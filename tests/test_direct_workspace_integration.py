import importlib.util
import pathlib
import sys


WORKSPACE = pathlib.Path("/Users/botwasabi/AI/ouroboros-workspace")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_prompt_contains_avoid_patterns():
    generator = _load(
        "direct_ad_generator_for_test",
        WORKSPACE / "skills" / "direct_ad_generator" / "generator.py",
    )
    prompt = generator._build_prompt(
        brief={"source": "test", "rows_count": 1, "hypotheses_to_generate": []},
        requests=[{"type": "scale_winner", "pattern": "winner", "evidence": {}}],
        offer="меню для похудения",
        platform_format="yandex_direct_search_text_ad",
        audience_awareness="warm",
        avoid_patterns=["too_generic", "Не повторять отклоненный заголовок: План питания для похудения"],
    )

    assert "Avoid patterns learned from student feedback" in prompt
    assert "too_generic" in prompt
    assert "План питания для похудения" in prompt


def test_workspace_ingestion_derives_impressions_from_ctr_real_headers():
    parser = _load(
        "direct_ingestion_parser_for_test",
        WORKSPACE / "skills" / "direct_ingestion" / "parser.py",
    )
    content = (
        "№ Кампании;Название кампании;Поисковый запрос;Заголовок;Текст;"
        "Расход, ₽;Клики;Конверсии;CR, %;CPA, ₽;CPC, ₽;CTR, %\n"
        "707365438;Кампания;запрос;Заголовок;Текст;471.00;100;3;3.00;157.00;4.71;2.50\n"
    )

    row = parser.parse_direct_export(content)[0]

    assert row.impressions == 4000
    assert row.cost == 471.0
    assert row.ctr == 0.025


def test_direct_pipeline_sample_export_returns_creative_brief():
    pipeline = _load(
        "direct_creative_loop_for_test",
        WORKSPACE / "pipelines" / "direct_creative_loop.py",
    )
    sample = WORKSPACE / "pipelines" / "sample_direct_export.csv"

    brief = pipeline.build_creative_brief_from_files([sample])

    assert brief.rows_count > 0
    assert brief.hypotheses_to_generate
    assert brief.to_dict()["source"]
