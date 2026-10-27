import unittest
from unittest.mock import patch

from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite, parse_spec_tp_sync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DRAFT = SpecTpSyncSite.DSPARK_DRAFT


class TestParseSpecTpSync(CustomTestCase):
    def test_retired_draft_handles_select_the_merged_site(self):
        for token in ("dspark-graph-greedy", "dspark-draft-sample", "2", "6"):
            with self.subTest(token=token):
                self.assertEqual(parse_spec_tp_sync(token), frozenset({DRAFT}))

    def test_negating_a_retired_handle_drops_the_merged_site(self):
        sites = parse_spec_tp_sync("all,-6")
        self.assertNotIn(DRAFT, sites)
        self.assertEqual(sites, frozenset(SpecTpSyncSite) - {DRAFT})

    def test_only_native_npu_draft_noise_needs_rng_sync(self):
        for npu in (False, True):
            with (
                self.subTest(npu=npu),
                patch("sglang.srt.speculative.spec_tp_sync.is_npu", return_value=npu),
            ):
                self.assertEqual(DRAFT in parse_spec_tp_sync("rng"), npu)
        self.assertIn(DRAFT, parse_spec_tp_sync("dspark-draft"))


if __name__ == "__main__":
    unittest.main()
