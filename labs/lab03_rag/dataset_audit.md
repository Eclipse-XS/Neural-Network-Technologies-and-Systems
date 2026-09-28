# Підготовка корпусу Lab 3: Redis Vector Similarity + CSV

## Джерело та межі роботи

Джерело, вказане користувачем: https://www.kaggle.com/datasets/anvitkumar/shopping-dataset. Користувач самостійно надав розпаковані дані в `labs/lab03_rag/data/`. Завантаження через Kaggle API агентом не виконувалося; автентифікація не налаштована. Версію Kaggle за локальним архівом не встановлено.

Локальний архів: `C:/Users/Eclipse/Downloads/archive.zip`. ZIP пройшов CRC-перевірку. У ньому 98 CSV; усі 98 наданих CSV побайтово збігаються з відповідними ZIP entries (SHA-256). Архів залишено на місці. Повний розпакований набір залишається в ігнорованому `data/`, поза експериментальним каталогом `data/raw/csv/`.

Виконано лише аудит і копіювання трьох оригінальних CSV. CSVLoader, embeddings, Redis, LLM та RAG не реалізовано. Дані не очищено, рядки не видалено, значення не змінено. Комітів і push немає.

## Структура джерела

97 категорій містять 1000 рядків сумарно; `Combined_dataset.csv` містить 1000 рядків. Combined не входить до корпусу. Усі 98 файлів мають однакову впорядковану схему з 24 колонок. Файли прочитано як UTF-8; строгий CSV-парсер не виявив некоректних рядків або відмінної кількості полів. Точних дублікатів рядків усередині кожного файла немає.

Точні назви колонок (спільні для всіх трьох відібраних файлів):

```text
product_id
title
product_description
rating
ratings_count
initial_price
discount
final_price
currency
images
delivery_options
product_details
breadcrumbs
product_specifications
amount_of_stars
what_customers_said
seller_name
sizes
videos
seller_information
variations
best_offer
more_offers
category
```

Схема верхнього рівня однакова, але назви характеристик усередині `product_specifications` залежать від категорії. `product_details` містить JSON-об’єкт із `description`, `material_and_care`, `size_and_fit`. JSON масиви й об’єкти зберігаються як текстові CSV-поля. `seller_information` — текст із символами редагування/маскування, а не JSON.

## Вибір трьох категорій

| Файл | Категорія | Рядків | Колонок | Точних дублікатів |
|---|---|---:|---:|---:|
| dresses.csv | dresses | 100 | 24 | 0 |
| sports-shoes.csv | sports-shoes | 51 | 24 | 0 |
| earrings.csv | earrings | 34 | 24 | 0 |

Разом 185 рядків; усі `product_id` у спільному корпусі унікальні.

- `dresses.csv`: 100 товарів, опис тканини, довжини, силуету, приводу використання. Підтримує пошук за змістом та порівняння багатьох схожих товарів.
- `sports-shoes.csv`: 51 товар, характеристики виду спорту, матеріалу, амортизації, підошви. Підтримує функціональні запити на кшталт взуття для бігу або ходьби.
- `earrings.csv`: 34 товари, метал, покриття, форма, застібка, тип прикраси. Додає виразно іншу категорію з корисними описами та числовими полями.

Вибір ґрунтується на фактичній наявності описів і характеристик, повноті основних полів та кількості рядків. Для порівняння перевірено також shirts (97), handbags (18), watches (15). Більшість інших категорій надто малі: наприклад, backpacks має 4 товари. Три обрані категорії достатньо різні, а спільна схема дозволяє один майбутній механізм серіалізації. Корпус придатний для навчального експерименту; це не масштабний benchmark.

## Типи й пропуски

У всіх трьох файлах `product_id`, `ratings_count` визначені pandas як int64; `rating`, `initial_price`, `discount` — float64. `final_price` є текстом зі знаком ₹, розділювачами тисяч і буквальними лапками. Решта колонок — текст; виняток: повністю порожній `what_customers_said` у sports-shoes та earrings автоматично визначено як float64. Це наслідок inference, не зміна змісту поля. Майбутній loader має явно визначати типи.

Нижче кількість пропусків pandas. У колонках, не наведених у таблиці, пропусків немає. Зокрема, назви, описи, ціни, рейтинги, JSON деталей/характеристик, ID та категорії заповнені. Порожні вкладені значення перевіряються окремо.

| Поле | dresses | sports-shoes | earrings |
|---|---:|---:|---:|
| discount | 8 | 23 | 1 |
| seller_information | 10 | 14 | 1 |
| seller_name | 10 | 14 | 1 |
| variations | 61 | 32 | 17 |
| videos | 90 | 21 | 30 |
| what_customers_said | 37 | 51 | 34 |

### Приклади: dresses.csv

| product_id | title | product_description | initial_price | discount | final_price (raw) | rating | ratings_count |
|---|---|---|---:|---:|---|---:|---:|
| 21664722 | Curvy Clan | Colourblocked A-Line Cotton Maxi Dress | 2999.0 | (порожньо) | "₹2,999.00" | 3.7 | 3 |
| 21426422 | Trendyol | Black Maxi Dress | 4399.0 | 45.0 | "₹2,419.00" | 4.1 | 10 |
| 21238004 | COVER STORY | Embellished Shoulder Straps Velvet Sheath Dress | 3490.0 | 40.0 | "₹2,094.00" | 4.8 | 24 |

### Приклади: sports-shoes.csv

| product_id | title | product_description | initial_price | discount | final_price (raw) | rating | ratings_count |
|---|---|---|---:|---:|---|---:|---:|
| 17745816 | Lancer | Men Navy Blue Textile Running Non-Marking Shoes | 1999.0 | 40.0 | "₹1,999.00" | 4.0 | 6 |
| 17631178 | Slazenger | Men White Woven Design Walking Shoes | 3599.0 | (порожньо) | "₹3,599.00" | 3.7 | 127 |
| 13686682 | Campus | Women White Mesh Walking Shoes | 799.0 | 6.0 | "₹751.00" | 4.1 | 452 |

### Приклади: earrings.csv

| product_id | title | product_description | initial_price | discount | final_price (raw) | rating | ratings_count |
|---|---|---|---:|---:|---|---:|---:|
| 15720652 | Rubans | Silver-Toned Dome Shaped Studs Earrings | 1480.0 | 76.0 | "₹355.00" | 3.8 | 21 |
| 15713414 | Crunchy Fashion | White Contemporary Studs Earrings | 1998.0 | 68.0 | "₹639.00" | 4.2 | 22 |
| 15750524 | Priyaasi | Rose Gold-Plated Dome Shaped Jhumkas | 7999.0 | 74.0 | "₹2,079.00" | 4.4 | 97 |

У всіх 185 відібраних рядках `currency` = INR.

## Якість та обмеження

| Перевірка | dresses | sports-shoes | earrings |
|---|---:|---:|---:|
| discount > 0, але final_price = initial_price | 7 | 8 | 1 |
| Відхилення від initial_price × (1 − discount/100) понад 1 INR | 18 | 9 | 2 |
| rating = 0 і ratings_count = 0 | 11 | 0 | 11 |
| Елементи specifications без specification_value | 52 | 0 | 2 |
| Елементи specifications зі значенням NA | 841 | 170 | 51 |

Числа для specifications — кількість вкладених елементів, не товарів. Нульовий рейтинг у 22 товарах супроводжується нульовою кількістю оцінок; його не слід трактувати як підтверджену погану оцінку.

Приклад суперечності: sports-shoes, product_id 17745816, initial_price=1999, discount=40, final_price="₹1,999.00". Загалом у 16 записах ненульова знижка співіснує з незміненою ціною. У 29 записах відхилення від формули знижки перевищує 1 INR. Це діагностичний критерій, а не доказ того, яке поле правильне. Не перераховувати final_price і не вважати порожній discount нульовим без окремого рішення.

Усі 185 final_price вдалося тимчасово розібрати для аудиту. Від’ємних цін, final_price понад initial_price, повторів product_id та символів заміни U+FFFD немає. У потрібних JSON-полях product_details, product_specifications, breadcrumbs і sizes помилок JSON немає. Вкладені NA, відсутні specification_value, HTML-сутності та порожні об’єкти пропозицій/варіацій потрібно врахувати в майбутній текстовій репрезентації. seller_information маскований і непридатний для змістовного пошуку.

Назва `title` у перевірених прикладах містить брендоподібне значення (Curvy Clan, Lancer, Rubans), а опис самого товару — `product_description`. Окремої колонки `brand` немає: не створювати її як нібито вихідне поле.

## План майбутньої репрезентації (не реалізовано)

Один вихідний рядок — один документ. Текст: `product_description`, `title`, `category`, `initial_price`, `final_price` (зі збереженням raw), `currency`, `discount` (або явне «не вказано»), `rating`, `ratings_count`; далі змістовні значення з `product_details` і пар `specification_name: specification_value` у `product_specifications`. За потреби включити `sizes` і непорожній `seller_name`. Вкладені JSON потрібно читати як структури, не як довгі рядки службового синтаксису.

Не включати в embedding URL з images/videos, маскований seller_information, порожні variations/best_offer та рекламні more_offers. Ці поля залишаються у raw CSV. Довжину майбутнього тексту слід перевірити за токенізатором заданої embedding-моделі; пріоритет — опис і релевантні характеристики, щоб довгий запис не втрачав їх при обрізанні.

Метадані: `source_file` і `row_id` (похідний нульовий індекс запису без заголовка, не номер фізичного рядка CSV), а також фактичні `product_id`, `category`, `title`, `currency`, `initial_price`, `discount`, `rating`, `ratings_count`, `seller_name`. Для final_price зберігати raw та окреме явно похідне числове значення після перевірки формату. Не вигадувати відсутній brand. Стабільне посилання: source_file + row_id + product_id.

Майбутні запити можуть поєднувати призначення, матеріали, категорію, рейтинг і ціну та порівнювати кілька знайдених рядків. Семантичний top-k не гарантує глобальний мінімум ціни й не замінює SQL-агрегацію. Висновки потрібно обмежувати отриманими записами. Суперечливі ціни відтворювати з явною вказівкою суперечності.

## Перевірка копій та структура

В `data/raw/csv/` рівно три CSV, повторно прочитані без помилок. Кількість рядків збігається з оригіналами; SHA-256 кожної копії збігається з вихідним файлом і ZIP entry.

```text
labs/lab03_rag/
  dataset_audit.md
  data/
    .gitkeep
    [98 оригінальних CSV і початковий README.md]
    audit/
      inventory.json
      selected_audit.json
    raw/
      csv/
        dresses.csv
        sports-shoes.csv
        earrings.csv
```

Повний набір та audit JSON локальні й ігноруються Git. Майбутній loader повинен читати лише `data/raw/csv/*.csv`, а не рекурсивно весь `data/`.

| Файл | SHA-256 |
|---|---|
| dresses.csv | `846451524f325f58a19005fe5c47392495e76bdb37b5c8ae28368785e48ab267` |
| sports-shoes.csv | `93a0d131ef117c48558d97727ce06f796e3e71f7d12bb9b8c07cdb810ab729f3` |
| earrings.csv | `1040c970376583daa70aab84b798b71d3c1fbd351192b04363ebd03359021583` |

## Інвентар усіх вихідних CSV

| Файл | Рядків |
|---|---:|
| backpacks.csv | 4 |
| bath-robe.csv | 1 |
| bath-towels.csv | 1 |
| bathroom-accessories.csv | 1 |
| bedsheets.csv | 11 |
| belts.csv | 1 |
| bodysuit.csv | 2 |
| boots.csv | 2 |
| boxers.csv | 1 |
| bra.csv | 13 |
| bracelet.csv | 4 |
| briefs.csv | 12 |
| camisoles.csv | 2 |
| candle-holders.csv | 1 |
| caps.csv | 2 |
| casual-shoes.csv | 23 |
| chair-cover.csv | 1 |
| clothing-set.csv | 2 |
| clutches.csv | 1 |
| co-ords.csv | 1 |
| Combined_dataset.csv | 1000 |
| curtains-and-sheers.csv | 2 |
| cutlery.csv | 1 |
| dinnerware.csv | 1 |
| dresses.csv | 100 |
| duffel-bag.csv | 2 |
| dungarees.csv | 1 |
| dupatta.csv | 11 |
| earrings.csv | 34 |
| ethnic-dresses.csv | 20 |
| face-moisturisers.csv | 2 |
| face-wash-and-cleanser.csv | 1 |
| flats.csv | 12 |
| flip-flops.csv | 15 |
| floor-mats--dhurries.csv | 5 |
| formal-shoes.csv | 12 |
| foundation.csv | 1 |
| gloves.csv | 1 |
| hair-accessory.csv | 2 |
| hair-brush-and-comb.csv | 1 |
| hair-cream-and-mask.csv | 1 |
| hair-masks.csv | 1 |
| handbags.csv | 18 |
| headphones.csv | 3 |
| heels.csv | 9 |
| home-fragrance-set.csv | 1 |
| innerwear-vests.csv | 2 |
| jackets.csv | 29 |
| jeans.csv | 57 |
| jumpsuit.csv | 1 |
| kurta-sets.csv | 22 |
| kurtas.csv | 20 |
| kurtis.csv | 1 |
| lipstick.csv | 1 |
| lounge-pants.csv | 1 |
| lounge-shorts.csv | 1 |
| lounge-tshirts.csv | 3 |
| mangalsutra.csv | 1 |
| mobile-accessories.csv | 1 |
| night-suits.csv | 3 |
| nightdress.csv | 4 |
| outdoor-masks.csv | 2 |
| palazzos.csv | 12 |
| perfume-and-body-mist.csv | 1 |
| pillow-covers.csv | 1 |
| ring.csv | 3 |
| runners.csv | 3 |
| sandals.csv | 6 |
| saree-blouse.csv | 2 |
| sarees.csv | 22 |
| scarves.csv | 2 |
| shampoo-and-conditioner.csv | 3 |
| shaving-essentials.csv | 1 |
| sherwani.csv | 5 |
| shirts.csv | 97 |
| shorts.csv | 21 |
| showpieces.csv | 1 |
| skirts.csv | 6 |
| sports-shoes.csv | 51 |
| sunglasses.csv | 1 |
| sweaters.csv | 34 |
| sweatshirts.csv | 20 |
| swim-tops.csv | 1 |
| table-covers.csv | 1 |
| thermal-bottoms.csv | 1 |
| tights.csv | 1 |
| tops.csv | 122 |
| towel-set.csv | 1 |
| track-pants.csv | 9 |
| trolley-bag.csv | 1 |
| trousers.csv | 6 |
| trunk.csv | 6 |
| tshirts.csv | 39 |
| tunics.csv | 4 |
| wall-art.csv | 1 |
| wallets.csv | 7 |
| watches.csv | 15 |
| yoga-mats.csv | 1 |
