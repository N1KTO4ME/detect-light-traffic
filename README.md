##  Описание
Тестовое задание от КГМК

Этот проект — система **обнаружения светофора и определения активного сигнала** с использованием **OpenCV**.
Алгоритм ищет раму светофора, выделяет лампы и определяет цвет активного сигнала.  
Результат можно просмотреть прямо в браузере и скачать обработанное изображение.  

##  Функциональность
1. Загрузка изображения через веб-страницу  
2. Автоматическое обнаружение светофора и ламп  
3. Определение активного сигнала (**RED / YELLOW / GREEN**)  
4. Отображение результата на картинке  
5. Возможность скачать обработанное изображение  

## Технологии
> **OpenCV** — обработка фото
> 
> **Python** — язык программирования

## Принципы работы
1. Загруженное изображение обрабатывается алгоритмом:  
   - выделяются контуры рам светофора
```  
   h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Маски для тёмной и светлой рамки
    _, mask_dark = cv2.threshold(blur, 65, 255, cv2.THRESH_BINARY_INV)  # тёмные области
    _, mask_light = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY)     # светлые области

    # Границы (на всякий случай) — чтобы выделить контуры краёв
    edges = cv2.Canny(blur, 60, 150)
    edges = cv2.dilate(edges, np.ones((6,6), np.uint8), iterations=1)

    mask = cv2.bitwise_or(mask_dark, mask_light)
    mask = cv2.bitwise_or(mask, edges)

    # Морфология
    ksize = max(5, int(min(w_img, h_img) / 120))  # подстраивается под разрешение
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
   - находятся круги (лампы) внутри рам
```
   x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w].copy()
    if roi_img.size == 0:
        return None

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # маски
    red1 = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(red1, red2)

    mask_yellow = cv2.inRange(hsv, np.array([14, 80, 60]), np.array([40, 255, 255]))
    mask_green  = cv2.inRange(hsv, np.array([36, 60, 50]), np.array([100, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    color_masks = [('RED', mask_red), ('YELLOW', mask_yellow), ('GREEN', mask_green)]

    found = []
    roi_area = max(1, w * h)
    min_area = max(30, int(roi_area * 0.0015))
```  
   - определяется преобладающий цвет
```
for color_name, mask in color_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            (cxf, cyf), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cxf), int(cyf), int(r)
            if r <= 2:
                continue

            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue
            circularity = 4 * np.pi * area / (per * per)
            # фильтруем по круговости (при необходимости поднять порог)
            if circularity < 0.35:
                continue
            # относительный размер круга
            if r > h * 0.6 or r < h * 0.03:
                continue

            # маска круга, чтобы оценить средний HSV внутри круга
            mask_circle = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask_circle, (cx, cy), max(1, int(r * 0.85)), 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask_circle)
            mean_h, mean_s, mean_v = mean_hsv[0], mean_hsv[1], mean_hsv[2]

            if mean_s < 50 or mean_v < 60:
                continue

            # окончательное определение цвета по mean_h (wrap-around учтён)
            detected = None
            if mean_h < 12 or mean_h > 160:
                detected = 'RED 🔴'
            elif 12 <= mean_h <= 40:
                detected = 'YELLOW 🟡'
            elif 36 <= mean_h <= 100:
                detected = 'GREEN 🟢'
            else:
                detected = color_name

            # запомним в координатах исходного изображения (не ROI)
            found.append({
                'cx': x + cx,
                'cy': y + cy,
                'r': r,
                'color': detected,
                'mean_hsv': mean_hsv,
                'v': mean_v,
                'circularity': circularity
            })
```  
2. Результат сохраняется как новое изображение.  
3. На веб-странице показывается картинка с выделенным светофором и подписью сигнала (см. в приложение).  

## Установка и запуск
1. **Клонируем репозиторий**
   ```bash
   git clone https://github.com/N1KTO4ME/detect-light-traffic.git
   ```
2. **Устанавливаем зависимости**
   ```bash
   pip install -r requirements.txt
   ```
3. **Запускаем проект** (файл app.py)
   ```bash
   python app.py
   ```
## Приложение
<img width="865" height="305" alt="изображение" src="https://github.com/user-attachments/assets/337f37bf-d6d1-4edb-8889-893f09d9511a" />
<img width="857" height="1050" alt="изображение" src="https://github.com/user-attachments/assets/f6172458-2c1a-41e1-a93d-2610b4c4de70" />
<img width="656" height="1227" alt="изображение" src="https://github.com/user-attachments/assets/c9b693ce-cb69-4bfc-b40c-7d76c1c4f8c8" />
