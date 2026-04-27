"""Static Telegram intake copy shared by server routing and tests."""

STUDENT_HELP_TEXT = (
    "Я на связи. Отправь ссылку на Google Drive с выгрузкой из рекламного кабинета "
    "и коротко допиши нишу, оффер, гео и цель. Я разберу CTR/CPC/CPA, предложу "
    "заголовки, тексты и идеи статичных креативов с объяснением, от каких данных "
    "отталкивался.\n\n"
    "Если что-то не подошло, напиши почему и как должно быть лучше: конверсионнее, "
    "человечнее, конкретнее. Этот фидбек будет сохраняться для следующих пакетов."
)
STUDENT_STATUS_TEXT = (
    "Бот на связи. Для работы отправь Google Drive ссылку на выгрузку и краткий контекст: "
    "ниша, оффер, гео, цель."
)
OWNER_ONLY_TEXT = "Эта команда доступна только владельцу в Web UI."
OWNER_COMMANDS = {
    "/panic",
    "/restart",
    "/review",
    "/evolve",
    "/bg",
    "/student_view",
    "/student_inject",
    "/append_owner_pref",
}
STUDENT_COMMANDS = {
    "/start",
    "/help",
    "/new",
    "/cancel",
    "/status",
    "/take",
    "/skip",
    "/rewrite",
    "/more",
    "/done",
    "/delete_current",
}
STUDENT_AGENT_PREFIX = (
    "External Telegram student/media-buyer request. Treat this as client-facing "
    "Yandex Direct creative work, not an owner/self-modification chat. Do not reveal "
    "internal identity, code, commits, budget, rescue state, memory paths, or system "
    "diagnostics. If a Google Drive URL is present, use the read_url capability when "
    "available, ingest only accessible files, then use the Direct ingestion/analysis/"
    "generator/judge workflow when relevant. Ask only for missing campaign basics "
    "(niche, offer, geo, goal) if the export alone is insufficient. Return concise "
    "promo-material output and explain metric grounding."
)
