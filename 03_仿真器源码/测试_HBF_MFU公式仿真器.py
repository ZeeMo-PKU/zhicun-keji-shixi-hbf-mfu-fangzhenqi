import unittest

from HBF_MFU公式仿真器 import (
    Hardware,
    InputError,
    KV_WRITE_MIN_SECONDS,
    LAYER_COUNT,
    Scenario,
    build_workload,
    matrix_multiply_ops,
    simulate,
    simulate_layer,
)


def hardware(**overrides):
    values = {
        "config_name": "测试配置",
        "chip_num": 8,
        "mat_flops": 64e12,
        "util": 0.6,
        "hbf_bw": 256e9,
        "hbf_w_bw": 1e15,
        "hbf_lat": 1e-6,
        "ddr_bw": 30e9,
        "pcie_bw": 16e9,
        "pcie_lat": 5.4e-6,
    }
    values.update(overrides)
    return Hardware(**values)


class FormulaSimulatorTests(unittest.TestCase):
    def test_matrix_ops_count_multiplications_and_additions(self):
        self.assertEqual(matrix_multiply_ops(2, 3, 4), 2 * 4 * (2 * 3 - 1))

    def test_tail_moves_bytes_from_hbf_to_ddr_without_changing_history_size(self):
        hw = hardware()
        no_tail = build_workload(hw, Scenario("decode", 2, 1, 1024, 0))
        with_tail = build_workload(hw, Scenario("decode", 2, 1, 1024, 16))
        self.assertEqual(
            no_tail.p_bulk_kv,
            with_tail.p_bulk_kv + with_tail.p_tail_kv,
        )
        self.assertGreater(with_tail.p_tail_kv, 0)

    def test_ddr_path_waits_for_write_and_reads_tail(self):
        hw = hardware()
        work = build_workload(hw, Scenario("decode", 1, 1, 1024, 16))
        trace = simulate_layer(0.0, hw, work)
        self.assertGreaterEqual(trace["kv_ddr_load_s"], trace["newkv_ddr_write_e"])
        expected = (work.p_newkv + work.p_tail_kv) / hw.ddr_bw
        self.assertAlmostEqual(
            trace["kv_ddr_load_e"] - trace["kv_ddr_load_s"], expected
        )

    def test_scatter_reduce_uses_output_bytes(self):
        hw = hardware()
        work = build_workload(hw, Scenario("decode", 1, 1, 0, 0))
        trace = simulate_layer(0.0, hw, work)
        expected = hw.pcie_lat + work.p_ffn_out / hw.pcie_bw
        self.assertAlmostEqual(trace["scatterreduce_e"] - trace["ffn_e"], expected)

    def test_kv_write_obeys_minimum_duration(self):
        hw = hardware()
        work = build_workload(hw, Scenario("decode", 1, 1, 0, 0))
        trace = simulate_layer(0.0, hw, work)
        self.assertAlmostEqual(
            trace["kv_write_e"] - trace["kv_write_s"], KV_WRITE_MIN_SECONDS
        )

    def test_layers_are_chained(self):
        summary, traces, _ = simulate(
            hardware(), Scenario("prefill", 1, 16, 0, 0)
        )
        self.assertEqual(len(traces), LAYER_COUNT)
        self.assertAlmostEqual(traces[1]["layer_s"], traces[0]["layer_e"])
        self.assertAlmostEqual(summary["E2E latency(ms)"], traces[-1]["layer_e"] * 1000)

    def test_invalid_chip_count_is_rejected(self):
        with self.assertRaises(InputError):
            build_workload(hardware(chip_num=3), Scenario("decode", 1, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
