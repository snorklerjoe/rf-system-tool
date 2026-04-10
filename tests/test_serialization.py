"""
Unit tests for JSON serialisation (save_scene / load_scene) and
non-GUI export utilities (CSV, HTML report).
"""
import json
import os
import math
import tempfile
import pytest

from rf_tool.blocks.components import Amplifier, Attenuator, Source, Sink, Mixer
from rf_tool.serialization.json_io import save_scene, load_scene
from rf_tool.export.exporters import export_cascade_csv, export_html_report
from rf_tool.engine.cascade import compute_cascade_metrics


# ======================================================================= #
# JSON Save / Load                                                         #
# ======================================================================= #

class TestJsonSerialisation:
    def _make_blocks(self):
        src = Source(frequency=1e9, output_power_dbm=-10.0, snr_db=47.0, label="TX")
        amp = Amplifier(gain_db=20.0, nf_db=3.0, oip3_dbm=30.0, label="LNA")
        att = Attenuator(attenuation_db=6.0, label="6dB Pad")
        snk = Sink(label="RX")
        return [src, amp, att, snk]

    def _make_connections(self, blocks):
        return [
            {"src_block_id": blocks[0].block_id, "src_port": "OUT",
             "dst_block_id": blocks[1].block_id, "dst_port": "IN"},
            {"src_block_id": blocks[1].block_id, "src_port": "OUT",
             "dst_block_id": blocks[2].block_id, "dst_port": "IN"},
            {"src_block_id": blocks[2].block_id, "src_port": "OUT",
             "dst_block_id": blocks[3].block_id, "dst_port": "IN"},
        ]

    def test_save_creates_valid_json(self):
        blocks = self._make_blocks()
        conns = self._make_connections(blocks)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene(blocks, conns, filepath=path)
            with open(path) as fh:
                data = json.load(fh)
            assert "version" in data
            assert "blocks" in data
            assert "connections" in data
            assert len(data["blocks"]) == 4
            assert len(data["connections"]) == 3
        finally:
            os.unlink(path)

    def test_load_restores_blocks(self):
        blocks = self._make_blocks()
        conns = self._make_connections(blocks)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene(blocks, conns, filepath=path)
            result = load_scene(path)
            assert len(result["blocks"]) == 4
            loaded_ids = {b.block_id for b in result["blocks"]}
            original_ids = {b.block_id for b in blocks}
            assert loaded_ids == original_ids
        finally:
            os.unlink(path)

    def test_load_restores_block_types(self):
        blocks = self._make_blocks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene(blocks, [], filepath=path)
            result = load_scene(path)
            types = [type(b).__name__ for b in result["blocks"]]
            assert "Source" in types
            assert "Amplifier" in types
            assert "Attenuator" in types
            assert "Sink" in types
        finally:
            os.unlink(path)

    def test_load_restores_properties(self):
        amp = Amplifier(gain_db=17.5, nf_db=2.8, oip3_dbm=28.0, label="DUT")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene([amp], [], filepath=path)
            result = load_scene(path)
            loaded_amp = result["blocks"][0]
            assert loaded_amp.gain_db == pytest.approx(17.5)
            assert loaded_amp.nf_db == pytest.approx(2.8)
            assert loaded_amp.oip3_dbm == pytest.approx(28.0)
            assert loaded_amp.label == "DUT"
        finally:
            os.unlink(path)

    def test_load_restores_source_snr(self):
        src = Source(frequency=1.2e9, output_power_dbm=-5.0, snr_db=41.0, label="SRC")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene([src], [], filepath=path)
            result = load_scene(path)
            loaded_src = result["blocks"][0]
            assert isinstance(loaded_src, Source)
            assert loaded_src.snr_db == pytest.approx(41.0)
        finally:
            os.unlink(path)

    def test_load_restores_connections(self):
        blocks = self._make_blocks()
        conns = self._make_connections(blocks)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene(blocks, conns, filepath=path)
            result = load_scene(path)
            assert len(result["connections"]) == 3
        finally:
            os.unlink(path)

    def test_load_restores_annotations(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        annotations = [{"text": "Hello", "x": 10.0, "y": 20.0,
                         "font": "Arial", "font_size": 12, "color": "#FF0000"}]
        try:
            save_scene([], [], annotations=annotations, filepath=path)
            result = load_scene(path)
            assert len(result["annotations"]) == 1
            assert result["annotations"][0]["text"] == "Hello"
        finally:
            os.unlink(path)

    def test_save_with_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene([], [], filepath=path, metadata={"author": "Test", "version": "1.0"})
            with open(path) as fh:
                data = json.load(fh)
            assert data["metadata"]["author"] == "Test"
        finally:
            os.unlink(path)

    def test_roundtrip_mixer_preserves_lo_freq(self):
        mix = Mixer(lo_frequency=2.4e9)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_scene([mix], [], filepath=path)
            result = load_scene(path)
            from rf_tool.blocks.components import Mixer as MixerCls
            loaded = result["blocks"][0]
            assert isinstance(loaded, MixerCls)
            assert loaded.lo_frequency == pytest.approx(2.4e9)
        finally:
            os.unlink(path)


# ======================================================================= #
# CSV Export                                                               #
# ======================================================================= #

class TestCsvExport:
    def _make_metrics_and_blocks(self):
        blocks = [
            Amplifier(gain_db=20.0, nf_db=3.0, oip3_dbm=30.0, p1db_dbm=20.0, label="LNA"),
            Attenuator(attenuation_db=6.0, label="6dBPad"),
            Amplifier(gain_db=15.0, nf_db=5.0, oip3_dbm=25.0, label="PA"),
        ]
        metrics = compute_cascade_metrics(blocks)
        return metrics, blocks

    def test_csv_created(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_cascade_csv(metrics, blocks, filepath=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_csv_has_header(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_cascade_csv(metrics, blocks, filepath=path)
            with open(path) as fh:
                first_line = fh.readline()
            assert "Stage" in first_line
            assert "Gain" in first_line
        finally:
            os.unlink(path)

    def test_csv_has_correct_row_count(self):
        """Should have N stage rows + 1 header + 1 total row = N+2 lines."""
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_cascade_csv(metrics, blocks, filepath=path)
            with open(path) as fh:
                lines = fh.readlines()
            # header + 3 stages + 1 TOTAL = 5 lines
            assert len(lines) == len(blocks) + 2
        finally:
            os.unlink(path)

    def test_csv_total_row_present(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_cascade_csv(metrics, blocks, filepath=path)
            with open(path) as fh:
                content = fh.read()
            assert "TOTAL" in content
        finally:
            os.unlink(path)

    def test_csv_empty_blocks(self):
        metrics = compute_cascade_metrics([])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_cascade_csv(metrics, [], filepath=path)
            # Should have header + TOTAL = 2 lines
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 2
        finally:
            os.unlink(path)


# ======================================================================= #
# HTML Report Export                                                       #
# ======================================================================= #

class TestHtmlReport:
    def _make_metrics_and_blocks(self):
        blocks = [
            Amplifier(gain_db=20.0, nf_db=3.0, oip3_dbm=30.0, label="LNA"),
            Attenuator(attenuation_db=3.0, label="3dB Pad"),
        ]
        metrics = compute_cascade_metrics(blocks)
        return metrics, blocks

    def test_html_created(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_html_contains_doctype(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path)
            with open(path) as fh:
                content = fh.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)

    def test_html_contains_title(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path,
                               title="My Test Report")
            with open(path) as fh:
                content = fh.read()
            assert "My Test Report" in content
        finally:
            os.unlink(path)

    def test_html_contains_gain_value(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path)
            with open(path) as fh:
                content = fh.read()
            # Total gain = 20 - 3 = 17 dB
            assert "17.00" in content
        finally:
            os.unlink(path)

    def test_html_contains_block_labels(self):
        metrics, blocks = self._make_metrics_and_blocks()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path)
            with open(path) as fh:
                content = fh.read()
            assert "LNA" in content
            assert "3dB Pad" in content
        finally:
            os.unlink(path)

    def test_html_shows_na_for_none_values(self):
        """Blocks with no P1dB/OIP3 should show N/A in the report."""
        blocks = [Amplifier(gain_db=10.0, nf_db=3.0, p1db_dbm=None, oip3_dbm=None)]
        blocks[0].p1db_dbm = None
        blocks[0].oip3_dbm = None
        metrics = compute_cascade_metrics(blocks)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            export_html_report(metrics, blocks, filepath=path)
            with open(path) as fh:
                content = fh.read()
            assert "N/A" in content
        finally:
            os.unlink(path)
