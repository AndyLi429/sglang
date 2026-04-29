import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
META_PATH = (
    ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "linear"
    / "gdn_chunk_meta.py"
)


def _module() -> ast.Module:
    return ast.parse(META_PATH.read_text(encoding="utf-8"))


def _class(name: str) -> ast.ClassDef:
    for node in _module().body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _function(name: str) -> ast.FunctionDef:
    for node in _module().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class TestGDNChunkMetaStatic(unittest.TestCase):
    def test_metadata_exposes_all_prebuilt_gdn_chunk_fields(self):
        fields = {
            node.target.id
            for node in _class("GDNChunkedPrefillMetadata").body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue(
            {
                "chunk_indices_chunk64",
                "chunk_offsets_chunk64",
                "update_chunk_offsets_chunk64",
                "final_chunk_indices_chunk64",
                "chunk_indices_large_block",
                "block_indices_cumsum",
                "chunk_indices",
                "chunk_offsets",
            }.issubset(fields)
        )

    def test_builder_accepts_num_heads_for_cumsum_block_indices(self):
        fn = _function("build_gdn_chunked_prefill_meta")
        arg_names = {arg.arg for arg in [*fn.args.args, *fn.args.kwonlyargs]}
        self.assertIn("num_heads", arg_names)


if __name__ == "__main__":
    unittest.main()
