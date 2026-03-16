import argparse
import cv2
import numpy as np

CLASSES = ["person", "chair", "bag"]


def load_model(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers


def predict(net, out_layers, image, conf_thres=0.4, nms_thres=0.45, size=416):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf < conf_thres:
                continue

            cx, cy, bw, bh = det[:4]
            box_w = int(bw * w)
            box_h = int(bh * h)
            x = int(cx * w - box_w / 2)
            y = int(cy * h - box_h / 2)

            boxes.append([x, y, box_w, box_h])
            confidences.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((class_ids[i], confidences[i], boxes[i]))

    return results


def draw(image, detections):
    counts = {"person": 0, "chair": 0, "bag": 0}

    for class_id, conf, (x, y, w, h) in detections:
        class_name = CLASSES[class_id]
        counts[class_name] += 1

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{class_name} {conf:.2f}",
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    empty_est = max(counts["chair"] - counts["person"], 0)
    text = f"P:{counts['person']} C:{counts['chair']} B:{counts['bag']} EmptyEst:{empty_est}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    net, out_layers = load_model(args.cfg, args.weights)
    detections = predict(net, out_layers, image, args.conf, args.nms, args.imgsz)
    result = draw(image.copy(), detections)

    if args.save:
        cv2.imwrite(args.save, result)
        print(f"Saved: {args.save}")
    else:
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()