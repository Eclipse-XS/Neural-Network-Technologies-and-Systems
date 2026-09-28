"""Build evidence tables/figure from saved runs; prose refers to reviewed runs."""
import json
from .config import LAB_ROOT, OUTPUT_ROOT, MODEL_ID, MODEL_REVISION, GROUNDING_INSTRUCTION
from .evaluation import evaluated_results, factual_summary, table, markdown_table

THEORY = '''Мультимодальна модель використовує кілька типів даних та зв’язки між ними. У VQA зображення визначає доступні візуальні факти, а запитання визначає, які з них потрібні. Генерація відповіді залежить спільно від візуального та текстового входу. Captioning є загальним описом; тут він представлений лише описовою групою серед цільових питань.

Енкодер зображення утворює вектори ознак, проєктор узгоджує їх із простором мовних embeddings, tokenizer перетворює запитання на текстові токени. Авторегресивна мовна модель отримує обидва представлення і послідовно прогнозує токени відповіді. Це не просте з’єднання зображення і тексту як сирих даних. Загальна схема узгоджується з [документацією LLaVA](https://huggingface.co/docs/transformers/model_doc/llava).'''

ARCHITECTURE = '''```text
Зображення → SigLIP Vision → візуальні ознаки → MLP-проєктор ┐
                                                        ├→ Qwen2 → токени відповіді
Запитання → tokenizer → текстові embeddings ─────────────┘
```

За фактичним config: SigLIP Vision має 26 шарів, hidden_size=1152, patch_size=14, image_size=384; вибрано останній шар та full visual features. Проєктор узгоджує ознаки з hidden_size=1024 мовної частини. Qwen2 має 24 шари; text_config посилається на Qwen/Qwen1.5-0.5B-Chat. Зображення 512×512 обробляє штатний SiglipImageProcessor до 384×384. Ручне масштабування перед inference не застосовується.'''

RUBRIC = '''Оцінювання виконане асистентом вручну після перегляду зображення й кожної відповіді; це не незалежна експертна розмітка. Для фактів: CORRECT — правильна відповідь, PARTIALLY_CORRECT — частково правильна, INCORRECT — суперечить зображенню. Для описів: GROUNDED — підтверджений пікселями, MOSTLY_GROUNDED — переважно підтверджений із неточностями, HALLUCINATED_DETAIL — містить непідтверджену деталь. Для аналізу: SUPPORTED_BY_VISIBLE_EVIDENCE, PARTLY_SUPPORTED, UNSUPPORTED_SPECULATION розрізняють наявність конкретних візуальних доказів. Для контролю: CORRECT_UNCERTAINTY або HALLUCINATED. Неповнота відповіді та галюцинація — різні недоліки.

Точність підраховується тільки для трьох фактологічних baseline-питань. Контрольне питання, описові й аналітичні відповіді до цього знаменника не входять. У report/manual_evaluation.json збережено зв’язок оцінки з run_id, дослівною відповіддю і SHA зображення; зміна відповіді унеможливлює непомітне повторне використання старої оцінки.'''

ANALYSIS = {
    'descriptive': 'Модель розпізнала будинок, дерева, водойму та відбиття. В desc_01 непідтверджено названо матеріал даху wooden, і замість двох речень отримано три. Desc_02 короткий, але підтверджений. Desc_03 пропускає запитані просторові деталі й неточно каже on a lake замість біля берега.',
    'factual': 'Baseline: CORRECT=3, PARTIALLY_CORRECT=0, INCORRECT=0. Відповіді left, Brown, Yes відповідають видимим фактам. Це три прості цільові питання до одного кадру; результат не оцінює загальну точність LLaVA.',
    'analytical': 'Analysis_02 підкріплює природний характер сцени видимими об’єктами. Analysis_01 повторює висновок calm замість доказу. Analysis_03 правильно називає відбиття, але не пояснює перевернуті форми чи відповідність деревам над берегом. Отже, правильна категорія відповіді ще не означає якісного міркування.',
    'control': 'На природне запитання про точні географічні координати модель відповіла [0.542,0.544,0.856,0.731]. Чотири числа не є обґрунтованою відповіддю про місце; їх значення невідоме. За заданою рубрикою це HALLUCINATED: модель не визнала відсутність інформації попри спільну grounding-інструкцію. Числа нагадують рамку об’єкта, але таке походження не встановлено.',
    'temperature': 'За 0.2, 0.7 і 1.0 відповідь семантично стабільна: вода спокійна, видима опора — відбиття будинку й дерев. Довжини становлять 19, 23 і 23 токени включно з EOS; формулювання відрізняються, нових непідтверджених фактів немає. Погіршення за більшої температури тут не спостерігається. По одному seed на умову не дозволяє оцінити розподіл відповідей або загальну частоту галюцинацій. Baseline з greedy не входить до температурного sweep, бо відрізняється режим декодування.',
    'max_tokens': 'За 24 токени продовження обірвано на The cabin, EOS відсутній. За лімітів 64 і 128 отримано дослівно однакову завершену відповідь із 48 токенів включно з EOS. Додатковий бюджет не усунув непідтверджений матеріал даху. max_new_tokens є верхньою межею довжини, не бажаною довжиною і не гарантією якості. Рядок 64 повторно використовує baseline desc_01.',
    'paraphrase': 'Три рівнозначні запитання про горизонтальне положення дали left, center, Left. Перше й третє семантично узгоджені й правильні; друге хибне. Повної узгодженості немає, хоча всі параметри greedy однакові. Зміна регістру не є відмінністю змісту. Перший рядок повторно використовує baseline fact_01.',
}

LIMITATIONS = '''Один синтетичний кадр, три прості факти та одна реалізація sampling для кожної температури — малий навчальний експеримент. Обробка до 384×384 може втрачати дрібні деталі. Невелика мовна основа та мультимодальне узгодження не гарантують правильних просторових відповідей чи аргументації. Ручні якісні оцінки залежать від інтерпретації; оцінка матеріалу даху свідомо консервативна. Точний час, координати, люди поза кадром і фактичне походження сцени не доступні моделі з пікселів.

Однаковий seed скидається перед кожним sampling-викликом, але версії бібліотек, GPU, числова точність та реалізація kernels можуть змінювати результати. Час виміряно з CUDA synchronization; завантаження моделі виключено. Перший fact_01 включає прогрівання. Окремі затримки не є benchmark швидкодії. Веб використовувався лише для перевірки документації та завантаження ваг, а не для відповідей VQA.'''

CONCLUSIONS = '''LLaVA розпізнає головний зміст цього зображення та відповідає правильно на три базові факти, але аналітичні пояснення часто поверхові. Контроль координат виявив непідтверджену числову відповідь, а перефразування змінило правильне left на хибне center. Отже, baseline 3/3 не означає надійності за іншого формулювання.

У температурному sweep усі три відповіді залишилися візуально обґрунтованими. Ліміт 24 токени спричинив обрив, 64 і 128 дали однаковий текст із природним завершенням. На цьому прикладі більший бюджет не покращив зміст. Компактна модель придатна для демонстрації мультимодального inference на GPU 4 ГБ, але її точні твердження та пояснення потребують перевірки за зображенням.'''


def build_figure(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subset = frame[frame.experiment == 'max_tokens']
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout='constrained')
    bars = ax.bar(subset.max_new_tokens.astype(str), subset.generated_token_count,
                  color=['#b44949', '#256d85', '#256d85'], width=.55)
    for bar, row in zip(bars, subset.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{row.generated_token_count}\n' + ('без EOS' if row.reached_token_limit else 'EOS'),
                ha='center', va='bottom')
    ax.set(title='Фактична довжина відповіді за різних лімітів',
           xlabel='max_new_tokens', ylabel='Токени продовження (включно з EOS)', ylim=(0, 64))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    path = OUTPUT_ROOT / 'figures/token_budget.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_report():
    rows = json.loads((OUTPUT_ROOT / 'metadata/manifest.json').read_text(encoding='utf-8'))
    frame = evaluated_results(rows)
    if (frame.evaluation_label == 'NOT_REVIEWED').any():
        raise ValueError('Review new answers before rebuilding the written report')
    env = json.loads((OUTPUT_ROOT / 'metadata/environment.json').read_text())
    model = json.loads((OUTPUT_ROOT / 'metadata/model.json').read_text())
    source = json.loads((LAB_ROOT / 'assets/input/source.json').read_text())
    build_figure(frame)
    sections = [
        ('1. Тема', 'Робота з мультимодальними моделями.'),
        ('2. Мета', 'Дослідити взаємодію візуального та текстового входів у LLaVA через Hugging Face Transformers, якість VQA, галюцинації, параметри генерації та узгодженість.'),
        ('3. Варіант', 'Варіант 1 — Visual Question Answering. Виконано 10 baseline-питань і контрольовані порівняння, разом 19 рядків та 17 унікальних inference. Навчання не проводилося.'),
        ('4. Теоретичні відомості', THEORY),
        ('5. Архітектура мультимодальної моделі', ARCHITECTURE),
        ('6. Роль компонентів LLaVA', 'SigLIP Vision виділяє ознаки, MLP-проєктор переводить їх у простір мовної моделі, Qwen2 генерує відповідь. Built-in chat template вставляє візуальний вхід у належні позиції. Файл source.json і початковий prompt Lab 4 не передаються моделі.'),
        ('7. Використана модель', f"`{MODEL_ID}`, revision `{MODEL_REVISION}`. Фактичний клас `{model['model_class']}`, pipeline `{model['pipeline_base_class']}` із підкласом для читання кількості згенерованих токенів та EOS. Облік не змінює forward чи decoding. Позначення 0.5B стосується мовної основи; у повній моделі виміряно **{model['parameter_count']:,} параметри**. FP16, cuda:0, SDPA, batch_size=1. Квантизація та offload не знадобилися. [Checkpoint](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf)."),
        ('8. Апаратне та програмне середовище', f"Windows; Python {env['python']}; PyTorch {env['versions']['torch']}; Transformers {env['versions']['transformers']}; Accelerate {env['versions']['accelerate']}; Pillow {env['versions']['Pillow']}; CUDA {env['cuda_runtime']}; torch.cuda.is_available()={env['cuda_available']}; {env['gpu_name']}, {env['vram_bytes']/2**30:.3f} GiB VRAM. Перед завантаженням моделі доступно {env['free_vram_bytes']/2**30:.2f} GiB VRAM, {env['free_ram_bytes']/2**30:.2f} GiB RAM і {env['free_disk_bytes']/2**30:.2f} GiB диска. Це миттєві значення, не вимоги до ресурсів. Model load: {model['model_load_seconds']:.3f} с; не включений у час відповідей. Пакети та CUDA не змінювалися. Повні версії збережено в outputs/metadata/environment.json."),
        ('9. Вхідне зображення', f"![Вхідне зображення](../assets/input/primary_image.png)\n\nprimary_image.png: {source['width']}×{source['height']}, PNG. Джерело: `{source['source_path_or_url']}`; скопійовано без зміни оригіналу. SHA-256: `{source['sha256']}`. Lab 5 використовує власну копію й не імпортує Lab 4. З пікселів видно будиночок ліворуч, коричневі дерев’яні стіни, темний дах, хвойні дерева, водойму та відбиття. Ground truth отримано з візуального огляду, а не prompt генератора."),
        ('10. Організація VQA-експерименту', 'Фіксований набір: 3 описові, 3 фактологічні, 3 аналітичні та 1 невідповідне зображенню контрольне питання. Запити англійською, пояснення українською. Спільна інструкція:\n\n> ' + GROUNDING_INSTRUCTION + '\n\nBaseline: do_sample=False, max_new_tokens=64, temperature/top_p/seed відсутні. Temperature: analysis_01, do_sample=True, top_p=0.9, max_new_tokens=64, seed=42 перед кожним запуском, temperature=0.2/0.7/1.0. Ліміт: desc_01, greedy, 24/64/128. Перефразування: fact_01, три рівнозначні формулювання, однаковий baseline.\n\nПерший fact_01 використано як smoke test і збережено в baseline; порожня чи нерозпізнана відповідь зупиняє виконання. Кожний результат атомарно записується в окремий JSON одразу після inference. SHA-підпис охоплює зображення, model/revision, текст питання, grounding, generation config та середовище. Manifest відділяє участь в експериментах від унікальних генерацій: повторно використано fact_01 та desc_01.\n\nКлючовий виклик із src/inference.py:\n\n```python\noutput = pipe(text=messages, return_full_text=False,\n              generate_kwargs=case.generation.kwargs())\n```\n\nПовний код: [model.py](../src/model.py), [inference.py](../src/inference.py), [experiments.py](../src/experiments.py), [run.py](../src/run.py). [Офіційний pipeline](https://huggingface.co/docs/transformers/tasks/image_text_to_text).'),
    ]
    for number, title, kind in [(11, 'Описові питання', 'descriptive'), (12, 'Фактологічні питання', 'factual'),
                                (13, 'Аналітичні питання', 'analytical'), (14, 'Контроль на галюцинації', 'unanswerable')]:
        subset = frame[(frame.experiment == 'baseline') & (frame.question_type == kind)]
        content = markdown_table(subset[['question_id', 'question', 'answer', 'evaluation_label', 'notes']])
        key = 'control' if kind == 'unanswerable' else kind
        sections.append((f'{number}. {title}', content + '\n\n' + ANALYSIS[key]))
    for number, title, experiment in [(15, 'Вплив temperature', 'temperature'),
                                       (16, 'Вплив max_new_tokens', 'max_tokens'),
                                       (17, 'Узгодженість перефразованих питань', 'paraphrase')]:
        content = markdown_table(table(frame, experiment)) + '\n\n' + ANALYSIS[experiment]
        if experiment == 'max_tokens':
            content += '\n\n![Фактична довжина відповіді](../outputs/figures/token_budget.png)'
        sections.append((f'{number}. {title}', content))
    sections += [
        ('18. Аналіз якості', RUBRIC + '\n\n' + markdown_table(factual_summary(frame).reset_index()) + '\n\nКількість токенів виміряно за фактичним continuation після input_ids; EOS зараховано, візуальні й текстові токени запиту виключено.'),
        ('19. Аналіз узгодженості', ANALYSIS['paraphrase'] + '\n\n' + ANALYSIS['temperature']),
        ('20. Обмеження', LIMITATIONS),
        ('21. Висновки', CONCLUSIONS),
    ]
    report = '# Лабораторна робота №5\n\n' + '\n\n'.join('## ' + title + '\n\n' + body for title, body in sections)
    report += '\n\n[Контрольні запитання](defense_notes.md) · [Аудит методики](methodology_audit.md) · [Скріншоти](screenshot_checklist.md)\n'
    (LAB_ROOT / 'report/report.md').write_text(report, encoding='utf-8')
    return frame


if __name__ == '__main__':
    build_report()
