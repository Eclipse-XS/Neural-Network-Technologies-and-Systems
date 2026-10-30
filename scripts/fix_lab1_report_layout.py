from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

REPORT = Path(r"C:\Users\Eclipse\Desktop\7 семестр\Нейромережеві технології і системи\max\ЛР-1_НТіС_Янкович_Максим_ОІ-43.docx")
LAB_ROOT = Path(r"C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems\max\lab1")
QA_DIR = Path(r"C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems\.codex_qa_lab1\audit_assets")


def replace_text(paragraph, text):
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def find_paragraph(document, prefix):
    return next(p for p in document.paragraphs if p.text.startswith(prefix))


def set_cell(cell, text):
    replace_text(cell.paragraphs[0], text)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def create_class_figure(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 3591, 1461
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = r"C:\Windows\Fonts\arial.ttf"
    bold_path = r"C:\Windows\Fonts\arialbd.ttf"
    title = ImageFont.truetype(bold_path, 62)
    label = ImageFont.truetype(font_path, 46)
    label_bold = ImageFont.truetype(bold_path, 46)
    small = ImageFont.truetype(font_path, 40)
    colors = ["#35669a", "#e56700"]
    counts = [11985, 11063]

    draw.text((260, 40), "Розподіл класів у Train split", font=title, fill="#222222")
    draw.text((2180, 40), "Співвідношення класів (%)", font=title, fill="#222222")
    left, top, right, bottom = 220, 230, 1960, 1240
    draw.line((left, bottom, right, bottom), fill="#777777", width=4)
    draw.line((left, top, left, bottom), fill="#777777", width=4)
    for value in range(0, 13001, 2000):
        y = bottom - int((value / 13000) * (bottom - top))
        draw.line((left, y, right, y), fill="#dddddd", width=3)
        draw.text((60, y - 25), f"{value}", font=small, fill="#333333")
    centers = [760, 1430]
    for center, count, color, name in zip(centers, counts, colors, ["ham", "spam"]):
        bar_height = int((count / 13000) * (bottom - top))
        draw.rectangle((center - 210, bottom - bar_height, center + 210, bottom), fill=color, outline="black", width=4)
        value_text = f"{count:,}"
        draw.text((center - 90, bottom - bar_height - 70), value_text, font=label_bold, fill="#222222")
        draw.text((center - 55, bottom + 35), name, font=label, fill="#222222")
    draw.text((720, 1360), "Цільовий клас", font=label_bold, fill="#222222")

    pie_box = (2260, 260, 3380, 1380)
    start = -90
    total = sum(counts)
    for count, color in zip(counts, colors):
        end = start + 360 * count / total
        draw.pieslice(pie_box, start=start, end=end, fill=color, outline="white", width=5)
        start = end
    draw.ellipse((2520, 520, 3120, 1120), fill="white")
    draw.text((2110, 760), "ham", font=label, fill="#222222")
    draw.text((3370, 760), "spam", font=label, fill="#222222")
    draw.text((2450, 720), "52.00%", font=label_bold, fill="white")
    draw.text((3010, 720), "48.00%", font=label_bold, fill="white")
    image.save(path)


def replace_media(docx_path, replacements):
    fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(docx_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(docx_path) as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                source = replacements.get(item.filename)
                zout.writestr(item, source.read_bytes() if source else zin.read(item.filename))
        os.replace(tmp_path, docx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


document = Document(REPORT)

updates = {
    "Сирий навчальний спліт": "Сирий навчальний спліт налічував 31 716 записів, тестовий — 2 000. У навчальних даних вилучено 51 порожній запис і 2 854 точних повтори, після чого 28 811 унікальних листів стратифіковано розділено (80% / 20%, seed = 42) на Train (23 048 листів: 11 985 ham, 11 063 spam) та Validation (5 763 листи: 2 997 ham, 2 766 spam). Початковий тестовий набір після очищення та повного усунення 334 дублікатів і перетинів із навчальними даними скорочено до 1 666 незалежних тестових прикладів (930 ham та 736 spam). Це виключає інформаційний витік.",
    "На рис. 1 наведено розподіл класів": "На рис. 1 наведено розподіл класів у очищеному Train split. Співвідношення становить 52,00% ham та 48,00% spam, тому класи є близькими за чисельністю, але не ідентичними.",
    "Рис. 1. Розподіл класів": "Рис. 1. Розподіл класів в очищеному Train split Enron Spam",
    "• Ham (клас 0)": "• Ham (клас 0): w_0 = 23048 / (2 · 11985) ≈ 0,9615;",
    "• Spam (клас 1)": "• Spam (клас 1): w_1 = 23048 / (2 · 11063) ≈ 1,0417.",
    "Це дозволяє збалансувати штрафи": "Це дещо збільшує штраф за помилки на менш чисельному класі spam і зменшує його для ham без штучного дублювання прикладів чи oversampling. Навчання проводилося оптимізатором AdamW (lr = 10⁻³, weight_decay = 10⁻⁴, batch_size = 64, gradient clipping = 1.0). Механізм Early Stopping було налаштовано з patience = 3 за валідаційним F1, але до завершення 12 епох умова дострокової зупинки не спрацювала. Контрольна точка зберігала стан із найкращим валідаційним F1.",
    "На рис. 6 зображено криві": "На рис. 6 зображено криві training/validation loss. Training loss послідовно зменшується, тоді як validation loss має загальну тенденцію до зниження, але коливається між епохами. Зокрема, після мінімуму 0,0567 на 11-й епосі значення зростає до 0,0659 на 12-й. Отже, крива не є монотонною; водночас масштаб коливань не вказує на критичне перенавчання.",
    "У PyTorch модуль nn.TransformerDecoderLayer": "У PyTorch модуль nn.TransformerDecoderLayer технічно містить блок cross-attention і очікує аргумент memory. У цій Decoder-Only реалізації memory передається як константний нульовий тензор torch.zeros(batch, 1, d_model), тому блок формально виконується, але не отримує представлень Encoder. Це відрізняє модель від канонічних decoder-only архітектур, у яких блок cross-attention фізично відсутній.",
    "На рис. 12 зображено динаміку навчання": "На рис. 12 зображено динаміку навчання: training loss зменшився з 5,8387 до 5,0044, а validation perplexity — з 208,78 до 159,00. Модель пройшла всі 6 запланованих епох; ранню зупинку в цьому циклі навчання не реалізовано. Найкращу контрольну точку за validation perplexity збережено на 6-й епосі. Зниження loss саме по собі не означає семантичної досконалості тексту.",
}

for prefix, text in updates.items():
    matches = [p for p in document.paragraphs if p.text.startswith(prefix)]
    if matches:
        replace_text(matches[0], text)

tokenizer_paragraph = find_paragraph(document, "У проєкті використано word-level токенізацію")
replace_text(tokenizer_paragraph, tokenizer_paragraph.text.replace("min_freq = 2", "min_freq = 3"))

table1 = document.tables[0]
set_cell(table1.cell(6, 1), "23 048 записів (11 985 ham, 11 063 spam)")
set_cell(table1.cell(7, 1), "5 763 записи (2 997 ham, 2 766 spam)")
set_cell(table1.cell(9, 1), "52,00% Ham / 48,00% Spam")
set_cell(table1.cell(10, 1), "10 000 токенів (min_freq = 3)")

for table_index, caption_prefix in [(3, "Таблиця 4."), (9, "Таблиця 10.")]:
    table = document.tables[table_index]
    find_paragraph(document, caption_prefix).paragraph_format.keep_with_next = True
    for row_index, row in enumerate(table.rows):
        prevent_row_split(row)
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
                paragraph.paragraph_format.keep_together = True

document.core_properties.author = "Янкович Максим"
document.core_properties.last_modified_by = "Янкович Максим"
document.save(REPORT)

fig1 = QA_DIR / "enron_class_distribution_corrected.png"
create_class_figure(fig1)
replace_media(REPORT, {
    "word/media/image2.png": fig1,
    "word/media/image4.png": LAB_ROOT / "outputs" / "figures" / "enron_split_distribution.png",
})
