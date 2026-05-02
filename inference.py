#!/usr/bin/env python3
"""
Helmet detection — laptop ↔ Raspberry Pi 5 streaming inference.

Architecture:
    laptop (--mode client)  →  TCP frames  →  Pi5 (--mode server)  →  TCP detections  →  laptop displays

The same script runs on both ends; only the --mode flag differs.

Pi5  : python inference.py --mode server --model best.onnx --port 5555
Laptop: python inference.py --mode client --host <PI5_IP> --port 5555 [--source 0]
"""

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Class names — order must match training data.yaml
CLASS_NAMES = ["with helmet", "without helmet", "rider", "number plate"]

# Per-class BGR colours for drawing
CLASS_COLORS = {
    0: (0, 200, 0),     # with helmet → green
    1: (0, 0, 220),     # without helmet → red
    2: (200, 200, 0),   # rider → cyan-ish
    3: (200, 0, 200),   # number plate → magenta
}

IMG_SIZE = 640          # model input size used during training
JPEG_QUALITY = 80       # frame compression quality for streaming
CONF_THRES = 0.40
IOU_THRES = 0.45


# ───────────────────────── socket framing helpers ──────────────────────────

def send_msg(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed message (4-byte big-endian length + payload)."""
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> bytes:
    """Receive a length-prefixed message."""
    (length,) = struct.unpack(">I", recv_exact(sock, 4))
    return recv_exact(sock, length)


# ─────────────────────── YOLOv5 pre/post-processing ────────────────────────

def letterbox(img: np.ndarray, new_shape: int = IMG_SIZE,
              color=(114, 114, 114)) -> tuple:
    """Resize + pad image to new_shape × new_shape, preserving aspect ratio.

    Returns padded image plus (ratio, dw, dh) needed to map detections back
    to the original frame coordinate system.
    """
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
    """Pure-NumPy non-maximum suppression. Returns indices to keep."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(raw: np.ndarray, ratio: float, dw: float, dh: float,
                orig_shape: tuple, conf_thres: float = CONF_THRES,
                iou_thres: float = IOU_THRES,
                keep_classes: set | None = None) -> list:
    """Convert raw YOLOv5 ONNX output to a list of detection dicts.

    raw shape: (1, N, 5 + num_classes) with N typically 25200 for 640 input.
    Each row: [cx, cy, w, h, obj, cls0, cls1, ...] in letterboxed coords.
    """
    pred = raw[0]
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    scores = obj[:, None] * cls_scores               # (N, num_classes)
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > conf_thres
    if not mask.any():
        return []
    boxes_xywh = pred[mask, :4]
    confs = max_scores[mask]
    classes = class_ids[mask]

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Class-aware NMS — run NMS per class.
    detections = []
    h0, w0 = orig_shape[:2]
    for c in np.unique(classes):
        if keep_classes is not None and int(c) not in keep_classes:
            continue
        cls_mask = classes == c
        keep = nms(boxes_xyxy[cls_mask], confs[cls_mask], iou_thres)
        for k in keep:
            cb = boxes_xyxy[cls_mask][k]
            # Reverse letterbox: undo padding, undo scale.
            x1 = (cb[0] - dw) / ratio
            y1 = (cb[1] - dh) / ratio
            x2 = (cb[2] - dw) / ratio
            y2 = (cb[3] - dh) / ratio
            x1 = max(0.0, min(w0 - 1, x1))
            y1 = max(0.0, min(h0 - 1, y1))
            x2 = max(0.0, min(w0 - 1, x2))
            y2 = max(0.0, min(h0 - 1, y2))
            detections.append({
                "cls": int(c),
                "name": CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(int(c)),
                "conf": float(confs[cls_mask][k]),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })
    return detections


# ───────────────────────────── Pi5 server ──────────────────────────────────

def run_server(model_path: str, host: str, port: int, keep_classes: set | None) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("onnxruntime not installed. Run setup.sh first.")

    if not Path(model_path).is_file():
        sys.exit(f"model not found: {model_path}\n"
                 f"Export with:  yolo export model=best.pt format=onnx imgsz=640  "
                 f"(on a machine with ultralytics installed)")

    print(f"[server] loading model: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4   # Pi5 has 4 Cortex-A76 cores
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path,
                                   sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"[server] input='{input_name}', shape={session.get_inputs()[0].shape}")

    # warm-up run — first inference is always slow
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("[server] warm-up complete")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    if keep_classes is not None:
        kept = ", ".join(f"{i}={CLASS_NAMES[i]}" for i in sorted(keep_classes) if i < len(CLASS_NAMES))
        print(f"[server] keeping classes: {kept}")
    print(f"[server] listening on {host}:{port}")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[server] client connected: {addr}")
            frame_count = 0
            t0 = time.time()
            try:
                while True:
                    payload = recv_msg(conn)                  # JPEG bytes
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        send_msg(conn, json.dumps({"error": "decode failed",
                                                   "detections": []}).encode())
                        continue

                    # preprocess
                    lb, r, dw, dh = letterbox(frame, IMG_SIZE)
                    img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = img.transpose(2, 0, 1)[None, ...]   # NCHW
                    img = np.ascontiguousarray(img)

                    t_inf = time.time()
                    raw = session.run(None, {input_name: img})[0]
                    inf_ms = (time.time() - t_inf) * 1000

                    dets = postprocess(raw, r, dw, dh, frame.shape, keep_classes=keep_classes)

                    response = {
                        "detections": dets,
                        "inference_ms": round(inf_ms, 1),
                    }
                    send_msg(conn, json.dumps(response).encode())

                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = frame_count / (time.time() - t0)
                        print(f"[server] {frame_count} frames, "
                              f"{fps:.1f} FPS avg, last inf={inf_ms:.0f}ms, "
                              f"dets={len(dets)}")
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as e:
                print(f"[server] client disconnected: {e}")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[server] shutting down")
    finally:
        srv.close()


# ─────────────────────────── laptop client ─────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        color = CLASS_COLORS.get(d["cls"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


def run_client(host: str, port: int, source: str, width: int) -> None:
    # Open video source (webcam index or file path)
    src: object = source
    if source.isdigit():
        src = int(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open video source: {source}")

    # Suggest a reasonable capture size to keep network usage modest.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 / 16))

    print(f"[client] connecting to {host}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
    except OSError as e:
        sys.exit(f"connection failed: {e}")
    print("[client] connected. Press 'q' or ESC to quit.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    t0 = time.time()
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[client] end of stream")
                break

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            send_msg(sock, buf.tobytes())

            resp = json.loads(recv_msg(sock).decode())
            detections = resp.get("detections", [])
            frame = draw_detections(frame, detections)

            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            inf_ms = resp.get("inference_ms", 0)
            overlay = f"FPS: {fps:.1f}   Pi5 inference: {inf_ms:.0f} ms   dets: {len(detections)}"
            cv2.putText(frame, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow("Helmet detection (Pi5 inference)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except (ConnectionError, BrokenPipeError) as e:
        print(f"[client] lost connection to server: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()


# ─────────────────────────────── entry ─────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["server", "client"],
                   help="server: run on Pi5 | client: run on laptop")
    p.add_argument("--host", default="0.0.0.0",
                   help="server: bind address (default 0.0.0.0) | "
                        "client: Pi5 IP address")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--model", default="best.onnx",
                   help="(server) path to ONNX model")
    p.add_argument("--source", default="0",
                   help="(client) webcam index (e.g. 0) or video file path")
    p.add_argument("--width", type=int, default=640,
                   help="(client) capture width hint")
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1],
                   help="(server) class IDs to keep (default: 0 1 — helmet / no helmet). "
                        "Pass --classes 0 1 2 3 to keep all four.")
    args = p.parse_args()

    if args.mode == "server":
        run_server(args.model, args.host, args.port, set(args.classes))
    else:
        if args.host in ("0.0.0.0", ""):
            sys.exit("--host must be the Pi5's IP address in client mode")
        run_client(args.host, args.port, args.source, args.width)


if __name__ == "__main__":
    main()
