import shutil
from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train16/weights/best.pt")

video_path = "testowa_tablica.mp4"
prefix = "testowanie_sieci/wyniki_filmu/"
raw_folder = prefix + video_path + "/raw"
boxed_folder = prefix + video_path + "/box"
labels_folder = prefix + video_path + "/labels"

id_to_label = {
    0: "car",
    1: "number_plate",
    2: "A",
    3: "B",
    4: "C",
    5: "D",
    6: "E",
    7: "F",
    8: "G",
    9: "H",
    10: "I",
    11: "J",
    12: "K",
    13: "L",
    14: "M",
    15: "N",
    16: "O",
    17: "P",
    18: "Q",
    19: "R",
    20: "S",
    21: "T",
    22: "U",
    23: "V",
    24: "W",
    25: "X",
    26: "Y",
    27: "Z",
    28: "1",
    29: "2",
    30: "3",
    31: "4",
    32: "5",
    33: "6",
    34: "7",
    35: "8",
    36: "9",
    37: "0"
}


def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def save_txt(labels, txt_path):
    with open(txt_path, "w") as f:
        for label in labels:
            f.write(label + "\n")


def video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    clear_folder(raw_folder)
    clear_folder(boxed_folder)
    clear_folder(labels_folder)

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(boxed_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()  # Zapisz oryginał

        if video_path != 0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (720, 1080))
            orig_frame = cv2.rotate(orig_frame, cv2.ROTATE_90_CLOCKWISE)
            orig_frame = cv2.resize(orig_frame, (720, 1080))

        results = model(frame)
        detected_labels = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 0, 255),
                              thickness=1)
                text = f"{class_id} {x1},{y1},{x2},{y2}"
                detected_labels.append(text)

                if class_id >= 1:
                    cv2.putText(frame,
                                class_name,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1)

        # Zapisywanie plików
        # cv2.imwrite(os.path.join(raw_folder, f"frame_{frame_id:06d}.png"), orig_frame)
        cv2.imwrite(os.path.join(boxed_folder, f"frame_{frame_id:06d}.png"), frame)
        save_txt(detected_labels, os.path.join(labels_folder, f"frame_{frame_id:06d}.txt"))

        frame_id += 1

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def map_labels_to_ids(expected_by_label):
    label_to_id = {v: k for k, v in id_to_label.items()}
    mapped = {}
    for label, count in expected_by_label.items():
        if label not in label_to_id:
            raise ValueError(f"Nieznana klasa: {label}")
        mapped[label_to_id[label]] = count
    return mapped


def verify_detections_with_counts(txt_folder, expectedClasses):
    expected_counts = map_labels_to_ids(expectedClasses)
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
    files_with_issues = []

    for fname in txt_files:
        path = os.path.join(txt_folder, fname)
        detection_counts = {k: 0 for k in expected_counts.keys()}

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if parts and parts[0].isdigit():
                    detected_id = int(parts[0])
                    if detected_id in detection_counts:
                        detection_counts[detected_id] += 1

        missing = []
        excess = []
        for class_id, required_count in expected_counts.items():
            count = detection_counts.get(class_id, 0)
            if count < required_count:
                missing.append(f"{id_to_label.get(class_id, class_id)} (wymagane: {required_count}, znalezione: {count})")
            elif count > required_count:
                excess.append(f"{id_to_label.get(class_id, class_id)} (wymagane: {required_count}, znalezione: {count})")

        if missing or excess:
            details = []
            if missing:
                details.append("Brakujące -> " + ", ".join(missing))
            if excess:
                details.append("Nadmierne -> " + ", ".join(excess))
            files_with_issues.append((fname, " | ".join(details)))

    total_files = len(txt_files)
    incorrect_files = len(files_with_issues)
    correct_files = total_files - incorrect_files

    if files_with_issues:
        print("Pliki z problemami wykryć:")
        for fname, problem_info in files_with_issues:
            print(f"{fname}: {problem_info}")
    else:
        print("Wszystkie pliki poprawne")

    print(f"\nStatystyki: \nPoprawnie: {correct_files}\nBłędnie: {incorrect_files}\nRazem: {total_files}")


if __name__ == "__main__":
    # video("assets/" + video_path)

    # Definiujemy oczekiwania przez podanie nazw klas i liczby obiektów
    expectedClasses = {
        "car": 0,
        "number_plate": 1,
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 1,
        "F": 0,
        "G": 1,
        "H": 0,
        "I": 0,
        "J": 0,
        "K": 2,
        "L": 0,
        "M": 0,
        "N": 0,
        "O": 0,
        "P": 0,
        "Q": 0,
        "R": 0,
        "S": 0,
        "T": 0,
        "U": 0,
        "V": 0,
        "W": 1,
        "X": 0,
        "Y": 0,
        "Z": 0,
        "1": 0,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "0": 0
    }
    verify_detections_with_counts(labels_folder, expectedClasses)
