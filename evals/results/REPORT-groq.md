# Evaluation report

- Date: 2026-11-22  |  Model: `qwen/qwen3.8-27b`
- **Routing accuracy: 15/15 (100%)**
- Calculator answers correct: 3/3

| Category | Correct |
|---|---|
| clear_en | 4/4 |
| clear_fa | 4/4 |
| multi | 3/3 |
| ambiguous | 1/1 |
| over_limit | 1/1 |
| distractor | 1/1 |
| no_skill_fa | 1/1 |

| # | Category | Prompt | Acceptable | Got | OK |
|---|---|---|---|---|---|
| 1 | clear_en | Can you condense this paragraph for me? Remote work has grow... | summarizer | summarizer | yes |
| 2 | clear_en | How would you say 'where is the train station' in German? | translator | translator | yes |
| 3 | clear_en | What's 18 multiplied by 47, minus 100? | calculator | calculator | yes |
| 4 | clear_en | Tell me a fun fact about octopuses. | general_chat / fallback | general_chat | yes |
| 5 | clear_fa | لطفاً این پاراگراف را کوتاه‌تر کن: کار از راه دور از سال ۲۰۲... | summarizer | summarizer | yes |
| 6 | clear_fa | این جمله رو برام به انگلیسی بگو: من هر روز صبح قبل از رفتن ب... | translator | translator | yes |
| 7 | clear_fa | نصف ۸۶۰ به علاوه ۱۵ می‌شود چند؟ | calculator | calculator | yes |
| 8 | clear_fa | امروز خیلی خسته‌ام، پیشنهادی برای استراحت داری؟ | general_chat / fallback | general_chat | yes |
| 9 | multi | Give me a short version of the following text, but in Persia... | summarizer>translator | summarizer>translator | yes |
| 10 | multi | این متن را اول به فارسی ترجمه کن و بعد خلاصه‌اش کن: Electric... | translator>summarizer | translator>summarizer | yes |
| 11 | multi | Translate 'thank you very much' to Persian and also tell me ... | translator+calculator | translator+calculator | yes |
| 12 | ambiguous | translate | clarify | clarify | yes |
| 13 | over_limit | Summarize the following text, translate the summary into Per... | summarizer>translator / summarizer+calculator / translator+calculator | summarizer>translator | yes |
| 14 | distractor | Summarize: The company earned 5 million dollars in 2023 and ... | summarizer | summarizer | yes |
| 15 | no_skill_fa | پایتخت استرالیا کدام شهر است؟ | general_chat / fallback | general_chat | yes |
