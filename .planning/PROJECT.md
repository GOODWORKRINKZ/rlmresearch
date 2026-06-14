# RLM Dev Assistant

## What This Is

RLM-сервер, использующий рекурсивные вызовы LLM для эффективной разработки ПО. Интегрируется с VS Code через OpenAI-совместимый API, позволяя Copilot использовать мощь рекурсивной обработки.

## Core Value

Сделать доступные модели (DeepSeek, Mimo) максимально продуктивными для разработки через парадигму RLM — вместо одного длинного промпта, LLM декомпозирует задачу, рекурсивно вызывает себя на частях, и собирает результат.

## Context

### Исследование
- MIT CSAIL бумага (arXiv:2512.24601v2, Jan 2026) — RLM outperforms GPT-5 на long-context задачах
- Официальная библиотека: `pip install rlms` (github.com/alexzhang13/rlm, 4.5k ⭐)
- RLM = LLM + REPL + Symbolic Recursion

### Доступные модели
- **DeepSeek V4 Pro**: $0.435/1M input (cache miss), $0.87/1M output. 1M context, 384K max output
- **DeepSeek V4 Flash**: $0.14/1M input (cache miss), $0.28/1M output. 1M context, 384K max output
- **Mimo V2.5 Pro**: OpenAI-совместимый API

### Ключевая архитектура RLM
- Root model (depth=0) обрабатывает основную логику
- `other_backends` для depth=1 sub-calls (ограничение: 1 дополнительный backend)
- Внутри REPL: `llm_query(prompt, model="name")` для маршрутизации по моделям
- `custom_tools` — инъекция функций/данных в REPL среду
- `persistent` — переиспользование окружения между вызовами
- `compaction` — авто-саммаризация истории

## Requirements

### Active

- [ ] RLM ядро с поддержкой DeepSeek API
- [ ] RLM ядро с поддержкой Mimo API
- [ ] OpenAI-совместимый API endpoint (FastAPI)
- [ ] VS Code Copilot интеграция через API
- [ ] Маршрутизация моделей по типу задачи (Pro для сложных, Flash для sub-calls)
- [ ] System prompt для dev-задач
- [ ] Custom tools: анализ кода, файловые операции
- [ ] RAG для индексации кодовой базы (ChromaDB)

### Out of Scope

- Fine-tuning моделей — используем существующие API
- Собственный IDE — интегрируемся в VS Code
- Веб-интерфейс — только API + CLI

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| DeepSeek V4 Pro для root calls | Лучшее качество reasoning | — Pending |
| DeepSeek V4 Flash для sub-calls | Дешевле в 3x, быстрее | — Pending |
| Mimo как альтернатива | Доступность, возможно лучше для某些задач | — Pending |
| FastAPI для API | Стандарт, async, OpenAI-совместимый | — Pending |
| ChromaDB для RAG | Простой, встраиваемый, хорош для кода | — Pending |
| arbitrary depth recursion | LLM сам решает когда рекурсировать | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-14 after initialization*
