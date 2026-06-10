import os
import csv
import time
import shutil
import base64
from io import BytesIO
from itertools import product

import numpy as np
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ==================================================
# CONFIG
# ==================================================

URL = "http://103.233.100.26:8080"

DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "labels.csv")

OVERWRITE_DATASET = True

IMG_SIZE = 512

# Jumlah sudut rotasi.
# 24 sudut = sekitar 15 derajat per sudut jika rotasinya stabil.
HORIZONTAL_STEPS = 24

# Besar drag untuk rotasi kamera.
HORIZONTAL_DRAG = 80

# Mode zoom.
# Pada OrbitControls / Three.js umumnya:
# deltaY positif  = zoom out
# deltaY negatif  = zoom in
# Jika hasilnya terbalik, cukup tukar nama "zoom_in" dan "zoom_out".
ZOOM_LEVELS = [
    {
        "zoom_id": 0,
        "zoom_name": "zoom_out",
        "wheel_delta": 300,
        "repeat": 6
    },
    {
        "zoom_id": 1,
        "zoom_name": "normal",
        "wheel_delta": 0,
        "repeat": 0
    },
    {
        "zoom_id": 2,
        "zoom_name": "zoom_in",
        "wheel_delta": -300,
        "repeat": 6
    },
]

# Test dulu dengan True.
# Setelah hasil cfg_00000 dicek dan semua tertutup, ubah ke False.
TEST_ONLY_NONE_CONFIG = False

PAGE_WAIT = 2
ANIMATION_WAIT = 1.2
ROTATE_WAIT = 0.35
ZOOM_WAIT = 0.8

BACKGROUND_COLOR = (32, 32, 32)


# ==================================================
# BUTTONS AND LABEL ORDER
# ==================================================

BUTTONS = [
    "Front Left Door",
    "Front Right Door",
    "Rear Left Door",
    "Rear Right Door",
    "Hood"
]

LABEL_COLUMNS = [
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "hood"
]


# ==================================================
# DATASET INIT
# ==================================================

def init_dataset():
    if OVERWRITE_DATASET and os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    os.makedirs(IMG_DIR, exist_ok=True)

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "front_left",
            "front_right",
            "rear_left",
            "rear_right",
            "hood",
            "config",
            "zoom_id",
            "zoom_name",
            "zoom_delta",
            "angle_id"
        ])


# ==================================================
# DRIVER INIT
# ==================================================

def init_driver():
    options = webdriver.ChromeOptions()

    # Ukuran window dibuat konsisten agar canvas stabil.
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")

    # Jangan aktifkan headless dulu saat debugging.
    # options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    print("Chrome terbuka.")
    print("Membuka URL:", URL)

    driver.get(URL)

    print("Current URL:", driver.current_url)

    wait_page_ready(driver)

    return driver


def wait_page_ready(driver, timeout=40):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.TAG_NAME, "canvas"))
    )

    time.sleep(PAGE_WAIT)


def reset_page(driver):
    driver.get(URL)
    wait_page_ready(driver)


# ==================================================
# CANVAS FUNCTIONS
# ==================================================

def get_canvas(driver):
    return driver.find_element(By.TAG_NAME, "canvas")


def get_canvas_rect(driver):
    rect = driver.execute_script("""
        const canvas = document.querySelector('canvas');
        const r = canvas.getBoundingClientRect();
        return {
            left: r.left,
            top: r.top,
            width: r.width,
            height: r.height
        };
    """)
    return rect


def get_canvas_image(driver):
    """
    Mengambil isi canvas sebagai PIL Image.
    Ini hanya isi canvas, bukan screenshot UI browser.
    """
    img_base64 = driver.execute_script("""
        const canvas = document.querySelector('canvas');
        return canvas.toDataURL('image/png').split(',')[1];
    """)

    png_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(png_data)).convert("RGB")
    return img


# ==================================================
# SAFE BACKGROUND POINT
# ==================================================

def estimate_bg_color(arr):
    """
    Estimasi warna background dari empat sudut gambar.
    """
    h, w, _ = arr.shape
    patch = max(10, min(h, w) // 20)

    corners = [
        arr[0:patch, 0:patch],
        arr[0:patch, w-patch:w],
        arr[h-patch:h, 0:patch],
        arr[h-patch:h, w-patch:w],
    ]

    bg_color = np.median(
        np.concatenate([c.reshape(-1, 3) for c in corners], axis=0),
        axis=0
    )

    return bg_color


def find_safe_background_point(img):
    """
    Mencari titik background aman agar drag/zoom tidak menekan mobil.
    Titik dikembalikan dalam koordinat image canvas.
    """
    arr = np.array(img).astype(float)
    h, w, _ = arr.shape

    bg_color = estimate_bg_color(arr)

    candidates = [
        (0.06, 0.06),
        (0.94, 0.06),
        (0.06, 0.94),
        (0.94, 0.94),
        (0.08, 0.50),
        (0.92, 0.50),
        (0.50, 0.08),
        (0.50, 0.92),
    ]

    patch_radius = max(8, min(h, w) // 40)

    best_point = None
    best_score = float("inf")

    for rx, ry in candidates:
        x = int(rx * w)
        y = int(ry * h)

        x1 = max(x - patch_radius, 0)
        y1 = max(y - patch_radius, 0)
        x2 = min(x + patch_radius, w)
        y2 = min(y + patch_radius, h)

        patch = arr[y1:y2, x1:x2]

        diff = np.abs(patch - bg_color)
        score = diff.max(axis=2).mean()

        if score < best_score:
            best_score = score
            best_point = (x, y)

    if best_point is None:
        best_point = (int(0.06 * w), int(0.06 * h))

    return best_point


def canvas_img_point_to_viewport(driver, x_img, y_img, img):
    """
    Mengubah koordinat gambar canvas menjadi koordinat viewport browser.
    """
    rect = get_canvas_rect(driver)

    img_w, img_h = img.size

    x_view = rect["left"] + (x_img / img_w) * rect["width"]
    y_view = rect["top"] + (y_img / img_h) * rect["height"]

    return int(x_view), int(y_view)


def get_safe_viewport_point(driver):
    img = get_canvas_image(driver)

    x_img, y_img = find_safe_background_point(img)
    x_view, y_view = canvas_img_point_to_viewport(driver, x_img, y_img, img)

    return x_view, y_view


# ==================================================
# BUTTON CONTROL
# ==================================================

def click_button(driver, text):
    xpath = f"//button[contains(normalize-space(.), '{text}')]"

    btn = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, xpath))
    )

    btn.click()
    time.sleep(ANIMATION_WAIT)


def set_configuration(driver, config):
    """
    config:
        (FL, FR, RL, RR, HOOD)

    Asumsi:
    Setelah halaman di-reset, semua pintu dan hood tertutup.
    """
    for state, btn_text in zip(config, BUTTONS):
        if state == 1:
            click_button(driver, btn_text)

    time.sleep(0.5)


# ==================================================
# CAMERA CONTROL
# ==================================================

def rotate_horizontal(driver, offset):
    """
    Drag dari area background agar tidak membuka pintu secara tidak sengaja.
    """
    x, y = get_safe_viewport_point(driver)

    driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
        "type": "mousePressed",
        "x": x,
        "y": y,
        "button": "left",
        "clickCount": 1
    })

    driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
        "type": "mouseMoved",
        "x": x + offset,
        "y": y,
        "button": "left",
        "buttons": 1
    })

    driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
        "type": "mouseReleased",
        "x": x + offset,
        "y": y,
        "button": "left",
        "clickCount": 1
    })

    time.sleep(ROTATE_WAIT)


def get_canvas_center_viewport(driver):
    """
    Mengambil titik tengah canvas dalam koordinat viewport.
    Untuk scroll/zoom, titik tengah lebih stabil daripada pojok.
    """
    rect = get_canvas_rect(driver)

    x = int(rect["left"] + rect["width"] / 2)
    y = int(rect["top"] + rect["height"] / 2)

    return x, y


def apply_zoom(driver, wheel_delta, repeat=1):
    """
    Scroll/zoom beberapa kali agar efek zoom terlihat jelas.

    Umumnya:
    deltaY positif  = zoom out
    deltaY negatif  = zoom in
    """
    if wheel_delta == 0 or repeat == 0:
        return

    # Untuk zoom, aman memakai tengah canvas karena tidak ada klik.
    x, y = get_canvas_center_viewport(driver)

    for _ in range(repeat):
        driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": x,
            "y": y,
            "deltaX": 0,
            "deltaY": wheel_delta
        })

        time.sleep(0.15)

    time.sleep(ZOOM_WAIT)


# ==================================================
# IMAGE PROCESSING
# ==================================================

def resize_with_letterbox(img, size=512, bg_color=BACKGROUND_COLOR):
    """
    Resize tanpa crop agar efek zoom tetap terlihat.

    Berbeda dengan auto-crop:
    - auto-crop membuat ukuran mobil selalu hampir sama;
    - letterbox mempertahankan skala mobil sesuai zoom asli.
    """
    img = img.convert("RGB")
    w, h = img.size

    scale = min(size / w, size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), bg_color)

    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def estimate_object_ratio(img, threshold=15):
    """
    Mengukur kira-kira seberapa besar objek non-background dalam gambar.
    Dipakai untuk audit apakah zoom in/out benar-benar berbeda.
    """
    arr = np.array(img).astype(float)
    bg = np.array(BACKGROUND_COLOR, dtype=float)

    diff = np.abs(arr - bg)
    mask = diff.max(axis=2) > threshold

    return float(mask.mean())


# ==================================================
# SAVE IMAGE
# ==================================================

def save_canvas(driver, config, zoom_info, angle_id):
    bits = "".join(map(str, config))

    zoom_id = zoom_info["zoom_id"]
    zoom_name = zoom_info["zoom_name"]
    zoom_delta = zoom_info["wheel_delta"]
    zoom_repeat = zoom_info["repeat"]

    filename = f"cfg_{bits}_{zoom_name}_a{angle_id:02d}.png"
    path = os.path.join(IMG_DIR, filename)

    img = get_canvas_image(driver)

    # Penting:
    # Jangan auto-crop, karena auto-crop menghapus efek zoom.
    img = resize_with_letterbox(img, size=IMG_SIZE)

    img.save(path)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            filename,
            config[0],
            config[1],
            config[2],
            config[3],
            config[4],
            bits,
            zoom_id,
            zoom_name,
            zoom_delta,
            angle_id
        ])

    return path


# ==================================================
# AUDIT DATASET
# ==================================================

def audit_dataset():
    import pandas as pd

    df = pd.read_csv(CSV_FILE)

    print("\n" + "=" * 60)
    print("AUDIT DATASET")
    print("=" * 60)

    print("Total row CSV       :", len(df))
    print("Unique image        :", df["image"].nunique())
    print("Total image files   :", len(os.listdir(IMG_DIR)))
    print("Duplicate image     :", df["image"].duplicated().sum())

    print("\nDistribusi label positif:")
    print(df[LABEL_COLUMNS].sum())

    print("\nProporsi label positif:")
    print(df[LABEL_COLUMNS].mean())

    print("\nJumlah data per zoom:")
    print(df["zoom_name"].value_counts())

    config_count = df.groupby(LABEL_COLUMNS).size().reset_index(name="count")

    print("\nJumlah konfigurasi unik:", len(config_count))
    print("\nFrekuensi kemunculan konfigurasi:")
    print(config_count["count"].value_counts().sort_index())

    if TEST_ONLY_NONE_CONFIG:
        expected_total = len(ZOOM_LEVELS) * HORIZONTAL_STEPS
    else:
        expected_total = 32 * len(ZOOM_LEVELS) * HORIZONTAL_STEPS

    print("\nExpected total image:", expected_total)

    if len(df) == expected_total:
        print("Status jumlah data  : OK")
    else:
        print("Status jumlah data  : TIDAK SESUAI")

    none_df = df[df[LABEL_COLUMNS].sum(axis=1) == 0]
    print("\nJumlah label none [0,0,0,0,0]:", len(none_df))

    print("=" * 60)


def audit_zoom_visual():
    """
    Audit sederhana untuk memastikan zoom_out, normal, zoom_in punya ukuran objek berbeda.
    Audit ini memakai config 00000 angle 00.
    """
    print("\n" + "=" * 60)
    print("AUDIT ZOOM VISUAL")
    print("=" * 60)

    sample_files = [
        "cfg_00000_zoom_out_a00.png",
        "cfg_00000_normal_a00.png",
        "cfg_00000_zoom_in_a00.png"
    ]

    for fname in sample_files:
        path = os.path.join(IMG_DIR, fname)

        if not os.path.exists(path):
            print(f"{fname}: tidak ditemukan")
            continue

        img = Image.open(path).convert("RGB")
        ratio = estimate_object_ratio(img)

        print(f"{fname}: object_ratio = {ratio:.4f}")

    print("\nCatatan:")
    print("- zoom_out seharusnya memiliki object_ratio paling kecil.")
    print("- zoom_in seharusnya memiliki object_ratio paling besar.")
    print("- Jika hasilnya terbalik, tukar nama zoom_in dan zoom_out.")
    print("=" * 60)


# ==================================================
# MAIN COLLECTION
# ==================================================

def main():
    init_dataset()

    driver = init_driver()

    try:
        if TEST_ONLY_NONE_CONFIG:
            all_configs = [(0, 0, 0, 0, 0)]
            print("MODE TEST: hanya mengambil config [0,0,0,0,0]")
        else:
            all_configs = list(product([0, 1], repeat=5))

        total_expected = len(all_configs) * len(ZOOM_LEVELS) * HORIZONTAL_STEPS

        print(f"Total Configurations : {len(all_configs)}")
        print(f"Zoom Levels          : {len(ZOOM_LEVELS)}")
        print(f"Horizontal Steps     : {HORIZONTAL_STEPS}")
        print(f"Expected Images      : {total_expected}")

        counter = 0

        for cfg_idx, config in enumerate(all_configs):
            bits = "".join(map(str, config))

            print("\n" + "-" * 60)
            print(f"[{cfg_idx + 1}/{len(all_configs)}] Config = {config} | bits={bits}")
            print("-" * 60)

            for zoom_info in ZOOM_LEVELS:
                zoom_name = zoom_info["zoom_name"]
                zoom_delta = zoom_info["wheel_delta"]
                zoom_repeat = zoom_info["repeat"]

                print(f"  Zoom: {zoom_name} | delta={zoom_delta} | repeat={zoom_repeat}")

                # Reset halaman agar kondisi awal kembali tertutup.
                reset_page(driver)

                # Set pintu/hood sesuai label.
                set_configuration(driver, config)

                # Terapkan zoom.
                apply_zoom(driver, zoom_delta, zoom_repeat)

                for angle_id in range(HORIZONTAL_STEPS):
                    save_canvas(driver, config, zoom_info, angle_id)
                    counter += 1

                    print(
                        f"\r    Saved: {counter}/{total_expected} "
                        f"| zoom={zoom_name} | angle={angle_id:02d}",
                        end=""
                    )

                    rotate_horizontal(driver, HORIZONTAL_DRAG)

                print()

        print("\nFinished.")
        print(f"Total Images Saved = {counter}")

    finally:
        driver.quit()

    audit_dataset()
    audit_zoom_visual()


if __name__ == "__main__":
    main()