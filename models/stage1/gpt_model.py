import copy

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.grad_updater import FSDPWeightGradManager
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import Add, PlaceHolder


# Phase 17 (KernelCache): link-dict resilience helper for inference mode.
# When `inference=True`, load_tensor_graph drops the backward subgraph,
# so any entry in a `links` dict that references a dropped tensor would
# blow up inside ConnectGraph.apply. This helper keeps only entries
# whose endpoints both exist in the union of graphs being connected,
# so callers don't need to enumerate which specific backward tensors
# to skip.
def _keep_existing_links(graphs, links):
    ids = set()
    for g in graphs:
        for t in g.tensors:
            ids.add(t.id)
    def _norm(name):
        return name if "@" in name else f"{name}@0"
    return {k: v for k, v in links.items() if _norm(k) in ids and _norm(v) in ids}


def group_query_attention(GQA_surrounding_path=None, GQA_kernel_path=None, tpsp=True, inference=False):
    if GQA_surrounding_path is None:
        GQA_surrounding_path = "./sharding_spreadsheets/module3/tpsp_gpt/group_query_attention_surrounding.csv"
        if not tpsp:
            GQA_surrounding_path = (
                "./sharding_spreadsheets/module3/tp_gpt/group_query_attention_surrounding.csv"
            )
    if GQA_kernel_path is None:
        # GQA_kernel_path = (
        #     "./sharding_spreadsheets/module3/group_query_attention_kernel.csv"
        # )
        GQA_kernel_path = "./sharding_spreadsheets/module3/tpsp_gpt/group_query_attention_kernel_fused.csv"
        if not tpsp:
            GQA_kernel_path = (
                "./sharding_spreadsheets/module3/tp_gpt/group_query_attention_kernel_fused.csv"
            )
    GQA_surrounding = TensorGraph.load_tensor_graph(GQA_surrounding_path, inference=inference)
    GQA_kernel = TensorGraph.load_tensor_graph(GQA_kernel_path, inference=inference)
    GQA_kernel = ReplicateGraph.apply(GQA_kernel, "attn_kernel.%s")
    links = dict()
    links["q"] = "attn_kernel.q"
    links["k"] = "attn_kernel.k"
    links["v"] = "attn_kernel.v"
    links["attn_kernel.dq"] = "dq"
    links["attn_kernel.dk"] = "dk"
    links["attn_kernel.dv"] = "dv"

    links["attn_kernel.qkv"] = "attn"
    links["dattn"] = "attn_kernel.dqkv"

    graphs = [GQA_surrounding, GQA_kernel]
    if inference:
        links = _keep_existing_links(graphs, links)
    GQA = ConnectGraph.apply(graphs, links)
    return GQA


def feed_forward_network(ffn_path=None, tpsp=True, inference=False):
    if ffn_path is None:
        ffn_path = (
            "./sharding_spreadsheets/module3/tpsp_gpt/llama_feed_forward_network.csv"
        )
        if not tpsp:
            ffn_path = (
                "./sharding_spreadsheets/module3/tp_gpt/llama_feed_forward_network.csv"
            )
    ffn = ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path, inference=inference), "ffn.%s")
    return ffn


def transformer_decoder_block(ffn_path=None, layernorm_path=None, residual_path=None, tpsp=True, inference=False):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module3/tpsp_gpt/layer_norm.csv"
        if not tpsp:
            layernorm_path = (
                "./sharding_spreadsheets/module3/tp_gpt/layer_norm.csv"
            )
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module3/tpsp_gpt/residual.csv"
        if not tpsp:
            residual_path = "./sharding_spreadsheets/module3/tp_gpt/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path, inference=inference), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(group_query_attention(tpsp=tpsp, inference=inference), "mha.%s")
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path, inference=inference), "mha_res.%s"
    )

    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path, inference=inference), "post_attn_norm.%s"
    )

    ffn = feed_forward_network(ffn_path, tpsp=tpsp, inference=inference)

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path, inference=inference), "ffn_res.%s"
    )

    links = dict()
    # input_layernorm
    links["input_norm.y"] = "mha.x"
    # links["mha_dx"] = "input_norm_dy"

    # mha
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.dx1"] = "mha.do"
    # links["mha_res_dx2"] = "input_norm_dy"

    # mha res
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.dx"] = "mha_res.dy"

    # post_layer_norm
    links["post_attn_norm.y"] = "ffn.x0"
    # links["ffn_dx0"] = "post_layer_norm_dy"

    # ffn
    links["ffn.xdown"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"
    links["ffn_res.dx1"] = "ffn.dxdown"
    # links["ffn_res_dx2"] = "post_layer_norm_dy"

    graphs = [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res]
    if inference:
        links = _keep_existing_links(graphs, links)
    decoder_block = ConnectGraph.apply(graphs, links)

    # Phase 17: the two Add-op rewrites below (input_norm.dy, post_attn_norm.dy)
    # and FSDPWeightGradManager manipulate backward tensors by id. Skip in
    # inference mode where those tensors no longer exist.
    if not inference:
        tensor_id_map_tensor = decoder_block.get_tensor_id_map_tensor()

        input_norm_dy = tensor_id_map_tensor["input_norm.dy@0"]
        assert input_norm_dy.op_type == PlaceHolder.type_name
        input_norm_dy.op_type = Add.type_name
        input_norm_dy.x1 = tensor_id_map_tensor["mha.dx@0"]
        input_norm_dy.x2 = tensor_id_map_tensor["mha_res.dx2@0"]
        input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
        input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
        decoder_block.in_tensors.remove(input_norm_dy)
        decoder_block.out_tensors.remove(input_norm_dy.x1)
        decoder_block.out_tensors.remove(input_norm_dy.x2)

        post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm.dy@0"]
        assert post_attn_norm_dy.op_type == PlaceHolder.type_name
        post_attn_norm_dy.op_type = Add.type_name
        post_attn_norm_dy.x1 = tensor_id_map_tensor["ffn.dx0@0"]
        post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
        post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
        post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
        decoder_block.in_tensors.remove(post_attn_norm_dy)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

        decoder_block = FSDPWeightGradManager.apply(decoder_block)

    return decoder_block


def transformer_decoders(num_layers, decoder_template, tpsp=True, inference=False):
    links = dict()
    decoders = list()
    for i in range(num_layers):
        decoder = ReplicateGraph.apply(decoder_template, f"transformer.{i}.%s")
        decoders.append(decoder)
        if i > 0:
            links[f"transformer.{i-1}.ffn_res.y"] = f"transformer.{i}.input_norm.x"
            links[f"transformer.{i}.input_norm.dx"] = f"transformer.{i-1}.ffn_res.dy"

    if inference:
        links = _keep_existing_links(decoders, links)
    decoders = ConnectGraph.apply(decoders, links)
    return decoders


def gpt(num_layers, embedding_path=None, regenerate=False, tpsp=True, inference=False):
    from . import CACHE_DIR
    import os

    # Phase 17: include mode in cache filename so a training cache doesn't
    # get served to an inference request (or vice versa).
    mode_tag = "inference" if inference else "training"
    cache_filename = os.path.join(CACHE_DIR, f"gpt_{num_layers}_{tpsp}_{mode_tag}.csv")
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module3/tpsp_gpt/embedding.csv"
        if not tpsp:
            embedding_path = (
                "./sharding_spreadsheets/module3/tp_gpt/embedding.csv"
            )
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path, inference=inference),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path, inference=inference),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal"},
    )

    decoder_template = transformer_decoder_block(tpsp=tpsp, inference=inference)
    decoders = transformer_decoders(num_layers, decoder_template, tpsp=tpsp, inference=inference)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links["transformer.0.input_norm.dx"] = "in_emb.dy"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    graphs = [decoders, in_emb, out_emb]
    if inference:
        links = _keep_existing_links(graphs, links)
    transformer = ConnectGraph.apply(graphs, links)

    # Phase 17: loss is a backward-only concept; skip the entire loss load
    # + connect block when inference. This also avoids trying to filter the
    # backward-only tensors out of loss.csv (every row is backward).
    if not inference:
        if tpsp:
            loss = ReplicateGraph.apply(
                TensorGraph.load_tensor_graph(
                    "./sharding_spreadsheets/module3/tpsp_gpt/loss.csv"
                ),
                "loss.%s",
            )
        else:
            loss = ReplicateGraph.apply(
                TensorGraph.load_tensor_graph(
                    "./sharding_spreadsheets/module3/tp_gpt/loss.csv"
                ),
                "loss.%s",
            )
        links = dict()
        links["out_emb.y"] = "loss.y"
        links["loss.dy"] = "out_emb.dy"
        transformer = ConnectGraph.apply([transformer, loss], links)

    transformer.save_tensor_graph(cache_filename)
    return transformer
