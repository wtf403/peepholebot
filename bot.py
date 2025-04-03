import os
import logging
import asyncio
import tempfile
import cv2
import numpy as np
import subprocess

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, Router
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    ReplyKeyboardRemove,
    FSInputFile,
    ForceReply,
)

load_dotenv()
TOKEN = os.getenv("TOKEN")

logging.basicConfig(level=logging.INFO)

router = Router()

processing_users = set()

KB = ForceReply(
    input_field_placeholder="Send me a video note 🪩",
    resize_keyboard=True,
)


@router.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Welcome!",
        reply_markup=KB,
    )


@router.message(F.video_note)
async def handle_video(message: Message):
    user_id = message.from_user.id
    if user_id in processing_users:
        await message.answer(
            "Another video note is currently being processed. Please wait."
        )
        return

    processing_users.add(user_id)
    processing_msg = None
    try:
        processing_msg = await message.answer(
            "⏳ Processing your video note...", reply_markup=ReplyKeyboardRemove()
        )

        video_note = message.video_note
        file_info = await message.bot.get_file(video_note.file_id)
        file_path = file_info.file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            input_filename = os.path.join(temp_dir, f"input_{user_id}.mp4")
            await message.bot.download_file(file_path, destination=input_filename)

            if not os.path.exists(input_filename):
                raise FileNotFoundError(
                    f"Failed to download video note to {input_filename}"
                )

            output_path = await process_video_with_peephole_effect(input_filename)

            if output_path and os.path.exists(output_path):
                await message.reply_video_note(
                    FSInputFile(output_path),
                    reply_markup=KB,
                )
            else:
                raise RuntimeError("Video processing failed to produce an output file.")

        if processing_msg:
            await processing_msg.delete()

    except Exception as e:
        logging.exception(f"Error processing video for user {user_id}: {e}")
        if processing_msg:
            try:
                await processing_msg.delete()
            except Exception as del_e:
                logging.error(f"Failed to delete processing message: {del_e}")
        await message.answer(
            "❌ Sorry, there was an error processing your video.\nPlease check logs or try again later.",
            reply_markup=KB,
        )
    finally:
        if user_id in processing_users:
            processing_users.remove(user_id)


async def process_video_with_peephole_effect(input_path):
    temp_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(temp_dir, f"{base_name}_processed.mp4")
    temp_output_path = os.path.join(temp_dir, f"{base_name}_temp.mp4")

    cascade_filename = "haarcascade_frontalface_default.xml"
    face_cascade = None
    cascade_path_to_try = cascade_filename

    if not os.path.exists(cascade_path_to_try):
        cv2_data_path = getattr(cv2, "data", None)
        haarcascades_path = (
            getattr(cv2_data_path, "haarcascades", None) if cv2_data_path else None
        )
        if haarcascades_path and os.path.exists(
            os.path.join(haarcascades_path, cascade_filename)
        ):
            cascade_path_to_try = os.path.join(haarcascades_path, cascade_filename)
            logging.info(
                f"Using cascade file found in cv2 data path: {cascade_path_to_try}"
            )
        else:
            cascade_path_to_try = None
            logging.warning(
                f"Could not find '{cascade_filename}'. Face detection disabled."
            )

    if cascade_path_to_try:
        face_cascade = cv2.CascadeClassifier(cascade_path_to_try)
        if face_cascade.empty():
            logging.error(
                f"Failed to load cascade classifier from '{cascade_path_to_try}'."
            )
            face_cascade = None
        else:
            logging.info(f"Loaded Haar cascade from: {cascade_path_to_try}")

    def _process_video():
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.warning(f"Invalid FPS ({fps}). Using default 30 FPS.")
            fps = 30

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(
                f"Video frame width ({width}) or height ({height}) is invalid."
            )

        orig_size = min(width, height)
        center_x = orig_size / 2.0
        center_y = orig_size / 2.0
        max_radius = orig_size / 2.0
        epsilon = 1e-6

        calibration_frames = 10
        initial_face_widths = []
        frames_read_for_calib = 0
        logging.info(f"Starting calibration phase ({calibration_frames} frames)...")
        while frames_read_for_calib < calibration_frames:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Could not read enough frames for calibration.")
                break
            frames_read_for_calib += 1

            if frame.shape[0] != frame.shape[1] or frame.shape[0] != orig_size:
                if frame.shape[0] != frame.shape[1]:
                    min_dim = min(frame.shape[0], frame.shape[1])
                    start_y = (frame.shape[0] - min_dim) // 2
                    start_x = (frame.shape[1] - min_dim) // 2
                    frame = frame[
                        start_y : start_y + min_dim, start_x : start_x + min_dim
                    ]
                if frame.shape[0] != orig_size:
                    frame = cv2.resize(frame, (orig_size, orig_size))

            if face_cascade:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda item: item[2] * item[3])
                    initial_face_widths.append(largest_face[2])

        initial_face_width_avg = (
            np.mean(initial_face_widths) if initial_face_widths else orig_size * 0.2
        )
        logging.info(
            f"Calibration complete. Average initial face width: {initial_face_width_avg:.2f} (from {len(initial_face_widths)} detections)"
        )

        cap.release()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot re-open video file {input_path} after calibration")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_output_path, fourcc, fps, (orig_size, orig_size), isColor=True
        )
        if not out.isOpened():
            cap.release()
            raise IOError(f"Cannot open video writer for {temp_output_path}")

        circle_mask_orig = np.zeros((orig_size, orig_size), dtype=np.uint8)
        cv2.circle(
            circle_mask_orig, (int(center_x), int(center_y)), int(max_radius), 255, -1
        )
        inv_circle_mask_orig = cv2.bitwise_not(circle_mask_orig)

        vignette_mask = np.zeros((orig_size, orig_size), dtype=np.float32)
        cv2.circle(
            vignette_mask, (int(center_x), int(center_y)), int(max_radius), 1.0, -1
        )
        blur_ksize_val = max(1, orig_size // 10)
        blur_ksize = blur_ksize_val + 1 if blur_ksize_val % 2 == 0 else blur_ksize_val
        vignette_mask = cv2.GaussianBlur(vignette_mask, (blur_ksize, blur_ksize), 0)
        vignette_mask_3ch = cv2.cvtColor(vignette_mask, cv2.COLOR_GRAY2BGR)

        x_coords, y_coords = np.meshgrid(
            np.arange(orig_size, dtype=np.float32),
            np.arange(orig_size, dtype=np.float32),
        )
        dx = x_coords - center_x
        dy = y_coords - center_y
        r_pixels = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r_norm = np.minimum(1.0, r_pixels / max_radius)

        background = np.zeros((orig_size, orig_size, 3), dtype=np.uint8)
        background.fill(20)

        frame_counter = 0
        smoothed_strength = 0.0
        last_valid_target_strength = 0.0
        alpha = 0.15
        alpha_decay = 0.03
        scale_factor = 0.9
        max_strength_cap = 0.9

        logging.info("Precomputing static edge compression map...")
        map_x_edge = np.zeros((orig_size, orig_size), np.float32)
        map_y_edge = np.zeros((orig_size, orig_size), np.float32)
        edge_center_radius_ratio = 0.6
        edge_center_radius = orig_size * edge_center_radius_ratio
        edge_strength_factor = 0.3

        for y in range(orig_size):
            for x in range(orig_size):
                current_r = r_pixels[y, x]
                current_theta = theta[y, x]

                if current_r <= edge_center_radius:
                    map_x_edge[y, x] = float(x)
                    map_y_edge[y, x] = float(y)
                else:
                    edge_t = (current_r - edge_center_radius) / (
                        max_radius - edge_center_radius
                    )
                    edge_t = np.clip(edge_t, 0.0, 1.0)
                    compression = edge_strength_factor * edge_t**3
                    r_new = current_r * (1.0 - compression * edge_t)
                    new_x = center_x + r_new * np.cos(current_theta)
                    new_y = center_y + r_new * np.sin(current_theta)
                    map_x_edge[y, x] = new_x
                    map_y_edge[y, x] = new_y
        logging.info("Edge compression map computed.")

        logging.info("Starting frame processing loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_counter == 0:
                    logging.error("Failed to read the first frame.")
                else:
                    logging.info(f"Finished processing {frame_counter} frames.")
                break

            frame_counter += 1

            if frame.shape[0] != frame.shape[1] or frame.shape[0] != orig_size:
                if frame.shape[0] != frame.shape[1]:
                    min_dim = min(frame.shape[0], frame.shape[1])
                    start_y = (frame.shape[0] - min_dim) // 2
                    start_x = (frame.shape[1] - min_dim) // 2
                    frame = frame[
                        start_y : start_y + min_dim, start_x : start_x + min_dim
                    ]
                if frame.shape[0] != orig_size:
                    frame = cv2.resize(frame, (orig_size, orig_size))

            target_strength = last_valid_target_strength
            face_detected_this_frame = False

            if face_cascade and initial_face_width_avg > epsilon:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) > 0:
                    largest_face = max(faces, key=lambda item: item[2] * item[3])
                    current_face_width = largest_face[2]
                    face_ratio = current_face_width / initial_face_width_avg
                    current_target = max(0, (face_ratio - 1.0)) * scale_factor
                    target_strength = min(current_target, max_strength_cap)

                    last_valid_target_strength = target_strength
                    face_detected_this_frame = True

            current_alpha = alpha if face_detected_this_frame else alpha_decay
            smoothed_strength = (
                current_alpha * target_strength
                + (1.0 - current_alpha) * smoothed_strength
            )

            distortion_effect = smoothed_strength * (1.0 - r_norm**2)
            scale = 1.0 + distortion_effect
            scale = np.maximum(scale, epsilon)

            map_x = center_x + dx / scale
            map_y = center_y + dy / scale

            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            distorted_frame = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            edge_compressed_frame = cv2.remap(
                distorted_frame,
                map_x_edge,
                map_y_edge,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            masked_result = cv2.bitwise_and(
                edge_compressed_frame, edge_compressed_frame, mask=circle_mask_orig
            )

            vignetted_float = cv2.multiply(
                masked_result.astype(np.float32), vignette_mask_3ch, dtype=cv2.CV_32F
            )
            vignetted_frame = np.clip(vignetted_float, 0, 255).astype(np.uint8)

            background_part = cv2.bitwise_and(
                background, background, mask=inv_circle_mask_orig
            )
            final_result = cv2.add(background_part, vignetted_frame)

            out.write(final_result)

        cap.release()
        out.release()
        logging.info("OpenCV processing finished.")

        logging.info("Starting FFmpeg merging/re-encoding...")
        final_output_file = None
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"Original input file for audio not found: {input_path}"
                )
            if not os.path.exists(temp_output_path):
                raise FileNotFoundError(
                    f"Temporary OpenCV output not found: {temp_output_path}"
                )

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_output_path,
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=640:640,format=yuv420p",
                "-crf",
                "23",
                "-maxrate",
                "1M",
                "-bufsize",
                "2M",
                "-movflags",
                "+faststart",
                "-preset",
                "medium",
                "-an",
                output_path,
            ]
            logging.debug(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(
                ffmpeg_cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if result.returncode != 0:
                logging.error(
                    f"FFmpeg error (code {result.returncode}):\\nSTDERR:\\n{result.stderr}\\nSTDOUT:\\n{result.stdout}"
                )
                if os.path.exists(temp_output_path):
                    logging.warning(
                        f"FFmpeg failed. Renaming {temp_output_path} to {output_path}"
                    )
                    os.rename(temp_output_path, output_path)
                    final_output_file = output_path
                else:
                    raise RuntimeError(
                        "FFmpeg processing failed and temporary video file is missing!"
                    )
            else:
                logging.info("FFmpeg processing successful.")
                final_output_file = output_path
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)

        except FileNotFoundError as fnf_err:
            logging.error(
                f"FFmpeg execution failed: {fnf_err}. Is FFmpeg installed and in PATH?"
            )
            if os.path.exists(temp_output_path):
                logging.warning(f"Using raw OpenCV output: {temp_output_path}")
                os.rename(temp_output_path, output_path)
                final_output_file = output_path
            else:
                raise RuntimeError(
                    f"FFmpeg/File not found and temporary video file {temp_output_path} is missing!"
                )
        except Exception as e:
            logging.exception(f"Unexpected error during FFmpeg execution: {e}")
            if os.path.exists(temp_output_path):
                logging.warning(
                    f"Using raw OpenCV output due to FFmpeg error: {temp_output_path}"
                )
                os.rename(temp_output_path, output_path)
                final_output_file = output_path
            else:
                raise RuntimeError(
                    f"FFmpeg processing failed unexpectedly and temporary video file {temp_output_path} is missing!"
                )

        return final_output_file

    processed_output_path = None
    try:
        processed_output_path = await asyncio.to_thread(_process_video)
        if not processed_output_path or not os.path.exists(processed_output_path):
            raise FileNotFoundError(
                f"Output file '{processed_output_path or output_path}' not found after processing thread."
            )
        logging.info(f"Processing complete. Final output file: {processed_output_path}")
        return processed_output_path

    except Exception as e:
        logging.exception(f"Exception in processing thread or final check: {e}")
        if (
            os.path.exists(temp_output_path)
            and temp_output_path != processed_output_path
        ):
            try:
                os.unlink(temp_output_path)
            except OSError as unlink_err:
                logging.error(
                    f"Error cleaning up temp file {temp_output_path}: {unlink_err}"
                )
        if os.path.exists(output_path) and output_path != processed_output_path:
            try:
                os.unlink(output_path)
            except OSError as unlink_err:
                logging.error(
                    f"Error cleaning up potentially incomplete output file {output_path}: {unlink_err}"
                )
        raise e


async def main():
    if not TOKEN:
        logging.critical(
            "Bot token not found. Please set the TOKEN environment variable."
        )
        return

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.getLogger("aiogram").setLevel(logging.INFO)

    bot = Bot(token=TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logging.info("Starting bot polling...")
    try:
        await bot.get_me()
        logging.info("Bot token is valid.")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except Exception as e:
        logging.critical(f"Bot polling failed: {e}", exc_info=True)
    finally:
        logging.info("Closing bot session...")
        await bot.session.close()
        logging.info("Bot session closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
