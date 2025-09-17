import cv2
import numpy as np

def detect_contours_tl(img, debug=False):
    """
    Надёжный поиск прямоугольной рамы светофора.
    Возвращает список (x,y,w,h).
    """
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

    # Морфология (размер ядра адаптивен)
    ksize = max(5, int(min(w_img, h_img) / 120))  # подстраивается под разрешение
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    img_area = w_img * h_img
    min_area = img_area * 0.0005   # гибкий минимум
    max_area = img_area * 0.5      # не брать сверхбольшие объекты

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Предпочтение контурам с 4 вершинами
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0 or w == 0:
                continue
            aspect = w / float(h)
            # Ожидаем вертикальную форму (узкая и высокая) — корректируй под свои фото
            if 0.15 < aspect < 0.9:
                # проверка заполнения прямоугольника (чтобы не брать сильно дырявые контуры)
                if area / (w * h) > 0.4:
                    rois.append((x, y, w, h))
                    continue

        # fallback: проверим boundingRect прямоугольность и соотношение сторон
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            continue
        aspect = w / float(h)
        if 0.12 < aspect < 1.0 and area / (w*h) > 0.35:
            rois.append((x, y, w, h))

    # отсортируем по y (сверху вниз) — полезно, если несколько
    rois = sorted(rois, key=lambda r: r[1])

    if debug:
        dbg = img.copy()
        for (x,y,w,h) in rois:
            cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("frame_mask", mask)
        cv2.imshow("frame_debug", dbg)
        cv2.waitKey(0)
        cv2.destroyWindow("frame_mask")
        cv2.destroyWindow("frame_debug")

    return rois


def detect_lamps_improved(image, roi, debug=False):
    """
    Находит лампы внутри ROI и возвращает ОДИН цвет (самый "сильный" по яркости).
    Если не найдено — возвращает None.
    """
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w].copy()
    if roi_img.size == 0:
        return None

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # маски (диапазоны можно подрегулировать под твою выборку)
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

            # требуем "светящесть" и насыщенность
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

    # если ничего не найдено — возвращаем None
    if not found:
        if debug:
            cv2.imshow("mask_red", mask_red); cv2.imshow("mask_yellow", mask_yellow); cv2.imshow("mask_green", mask_green)
            dbg = image.copy()
            cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow("roi_debug", dbg); cv2.waitKey(0); cv2.destroyAllWindows()
        return None

    # выбираем "лучший" — самый яркий по V (часто это реальный активный свет)
    found_sorted = sorted(found, key=lambda k: (k['v'], k['circularity']), reverse=True)
    best = found_sorted[0]
    if debug:
        dbg = image.copy()
        cv2.circle(dbg, (int(best['cx']), int(best['cy'])), int(best['r']), (0,255,0), 2)
        cv2.putText(dbg, best['color'], (int(best['cx'])-10, int(best['cy'])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("best_lamp", dbg); cv2.waitKey(0); cv2.destroyAllWindows()

    return best['color']


# Твоя функция показа окна оставляем как есть
def show_fixed_window(winname, img, window_w, window_h):
    h, w = img.shape[:2]
    scale = min(window_w / w, window_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    x_offset = (window_w - new_w) // 2
    y_offset = (window_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    cv2.imshow(winname, canvas)
