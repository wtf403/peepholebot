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
    cascade_path_to_try = cascade_filename  # Try local first

    if not os.path.exists(cascade_path_to_try):
        # Try standard cv2 path if local fails
        cv2_data_path = getattr(cv2, "data", None)
        if (
            cv2_data_path
            and cv2_data_path.haarcascades
            and os.path.exists(
                os.path.join(cv2_data_path.haarcascades, cascade_filename)
            )
        ):
            cascade_path_to_try = os.path.join(
                cv2_data_path.haarcascades, cascade_filename
            )
            logging.info(
                f"Using cascade file found in cv2 data path: {cascade_path_to_try}"
            )
        else:
            cascade_path_to_try = None
            logging.warning(
                f"Could not find '{cascade_filename}' locally or in cv2 data path. Face detection disabled."
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

        # Original frame dimensions (square for video notes)
        orig_size = min(width, height)

        # Expand factor - how much we'll initially enlarge the frame
        # to compensate for shrinking during lens effect
        expand_factor = 1.25  # 25% expansion (increased to ensure no borders)

        # Calculate expanded frame size
        expanded_size = int(orig_size * expand_factor)

        # Ensure expanded size is even (for video codec compatibility)
        if expanded_size % 2 != 0:
            expanded_size += 1

        # Centers for original and expanded
        center_orig_x = orig_size / 2.0
        center_orig_y = orig_size / 2.0
        center_expanded_x = expanded_size / 2.0
        center_expanded_y = expanded_size / 2.0

        # Radius calculations
        max_radius_orig = orig_size / 2.0
        max_radius_expanded = expanded_size / 2.0

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (orig_size, orig_size))
        if not out.isOpened():
            logging.warning("avc1 codec not available, falling back to mp4v.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (orig_size, orig_size))
            if not out.isOpened():
                cap.release()
                raise IOError(f"Cannot open video writer for {temp_output_path}")

        # --- Set up masks for original size ---
        # Create circular mask for the original size
        circle_mask_orig = np.zeros((orig_size, orig_size), dtype=np.uint8)
        cv2.circle(
            circle_mask_orig,
            (int(center_orig_x), int(center_orig_y)),
            int(max_radius_orig),
            255,
            -1,
        )
        inv_circle_mask_orig = cv2.bitwise_not(circle_mask_orig)

        # Create expanded circular mask (larger than needed to ensure no borders)
        circle_mask_expanded = np.zeros((expanded_size, expanded_size), dtype=np.uint8)
        cv2.circle(
            circle_mask_expanded,
            (int(center_expanded_x), int(center_expanded_y)),
            int(max_radius_expanded),
            255,
            -1,
        )

        # --- Create vignette mask ---
        vignette_mask = np.zeros((orig_size, orig_size), dtype=np.float32)
        cv2.circle(
            vignette_mask,
            (int(center_orig_x), int(center_orig_y)),
            int(max_radius_orig),
            1.0,
            -1,
        )
        blur_ksize_val = max(1, orig_size // 10)
        blur_ksize = blur_ksize_val + 1 if blur_ksize_val % 2 == 0 else blur_ksize_val
        vignette_mask = cv2.GaussianBlur(vignette_mask, (blur_ksize, blur_ksize), 0)
        vignette_mask_3ch = cv2.cvtColor(vignette_mask, cv2.COLOR_GRAY2BGR)

        # --- Create edge stretch maps (larger size) ---
        x_expanded, y_expanded = np.meshgrid(
            np.arange(expanded_size, dtype=np.float32),
            np.arange(expanded_size, dtype=np.float32),
        )

        # Calculate distances from center
        dx_expanded = x_expanded - center_expanded_x
        dy_expanded = y_expanded - center_expanded_y
        r_expanded = np.sqrt(dx_expanded**2 + dy_expanded**2)
        theta_expanded = np.arctan2(dy_expanded, dx_expanded)

        # Normalize distances (0 at center, 1 at max radius)
        r_norm_expanded = np.minimum(1.0, r_expanded / max_radius_expanded)

        # Create edge stretching map
        # Higher stretch factor means more aggressive stretching at edges
        edge_stretch = 0.25  # Increased for better compensation

        # Stretch increases quadratically from center to edges
        stretch_factor = 1.0 + edge_stretch * (r_norm_expanded**2)

        # Apply outward stretch
        stretch_map_x = center_expanded_x + dx_expanded * stretch_factor
        stretch_map_y = center_expanded_y + dy_expanded * stretch_factor

        # Background for final output
        background = np.zeros((orig_size, orig_size, 3), dtype=np.uint8)
        background.fill(20)  # Dark gray background

        # For face detection tracking
        frame_counter = 0
        last_known_good_strength = 0.0
        epsilon = 1e-6

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

            # --- Ensure frame is square and right size ---
            if frame.shape[0] != frame.shape[1] or frame.shape[0] != orig_size:
                # Center crop to square if needed
                if frame.shape[0] != frame.shape[1]:
                    min_dim = min(frame.shape[0], frame.shape[1])
                    start_y = (frame.shape[0] - min_dim) // 2
                    start_x = (frame.shape[1] - min_dim) // 2
                    frame = frame[
                        start_y : start_y + min_dim, start_x : start_x + min_dim
                    ]

                # Resize to orig_size if needed
                if frame.shape[0] != orig_size:
                    frame = cv2.resize(frame, (orig_size, orig_size))

            # --- Calculate lens strength based on face detection ---
            dynamic_lens_strength = 0.0
            face_detected_this_frame = False

            if face_cascade:
                # Extract just the circular area for face detection
                circular_area = cv2.bitwise_and(frame, frame, mask=circle_mask_orig)
                gray = cv2.cvtColor(circular_area, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )

                if len(faces) > 0:
                    largest_face = max(faces, key=lambda item: item[2] * item[3])
                    fx, fy, fw, fh = largest_face
                    relative_width = fw / orig_size
                    min_face_ratio = 0.08
                    max_face_ratio = 0.5
                    max_strength = 0.35

                    current_strength = 0.0
                    if relative_width < min_face_ratio:
                        current_strength = 0.0
                    elif relative_width > max_face_ratio:
                        current_strength = max_strength
                    else:
                        progress = (relative_width - min_face_ratio) / (
                            max_face_ratio - min_face_ratio
                        )
                        current_strength = max_strength * progress

                    dynamic_lens_strength = current_strength
                    last_known_good_strength = current_strength
                    face_detected_this_frame = True
                else:
                    dynamic_lens_strength = last_known_good_strength
                    face_detected_this_frame = False
            else:
                dynamic_lens_strength = 0.0
                face_detected_this_frame = False

            # --- Extract circular area and expand it ---
            # Extract only the circular area from the original frame
            circular_area = cv2.bitwise_and(frame, frame, mask=circle_mask_orig)

            # Create expanded canvas (black background)
            expanded_frame = np.zeros((expanded_size, expanded_size, 3), dtype=np.uint8)

            # Place the circular area in the center of expanded canvas
            # Calculate offsets for placing original in center of expanded
            offset_y = (expanded_size - orig_size) // 2
            offset_x = (expanded_size - orig_size) // 2

            # Copy the circular area to expanded frame
            expanded_frame[
                offset_y : offset_y + orig_size, offset_x : offset_x + orig_size
            ] = circular_area

            # --- Apply edge stretching to expanded frame ---
            stretched_frame = cv2.remap(
                expanded_frame,
                stretch_map_x,
                stretch_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            # --- Apply dynamic lens effect ---
            if dynamic_lens_strength > epsilon:
                # Create magnification map that's strongest in center
                dynamic_magnification = 1.0 + dynamic_lens_strength * (
                    1.0 - r_norm_expanded**2
                )
                dynamic_magnification = np.maximum(dynamic_magnification, epsilon)

                # Calculate source coordinates with inverse mapping
                src_x = center_expanded_x + dx_expanded / dynamic_magnification
                src_y = center_expanded_y + dy_expanded / dynamic_magnification

                # Apply lens effect
                magnified_frame = cv2.remap(
                    stretched_frame,
                    src_x.astype(np.float32),
                    src_y.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
            else:
                # No magnification
                magnified_frame = stretched_frame

            # --- Crop back to original size from center ---
            crop_start_y = offset_y
            crop_start_x = offset_x
            crop_end_y = crop_start_y + orig_size
            crop_end_x = crop_start_x + orig_size

            result_frame = magnified_frame[
                crop_start_y:crop_end_y, crop_start_x:crop_end_x
            ]

            # --- Apply circular mask and vignette ---
            # Re-apply circular mask to ensure clean edges
            masked_result = cv2.bitwise_and(
                result_frame, result_frame, mask=circle_mask_orig
            )

            # Apply vignette effect
            vignetted_float = cv2.multiply(
                masked_result.astype(np.float32), vignette_mask_3ch, dtype=cv2.CV_32F
            )
            vignetted_frame = np.clip(vignetted_float, 0, 255).astype(np.uint8)

            # Combine with background
            background_part = cv2.bitwise_and(
                background, background, mask=inv_circle_mask_orig
            )
            final_result = cv2.add(background_part, vignetted_frame)

            # --- Add lens strength debug value ---
            debug_text = f"{dynamic_lens_strength:.3f}"
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                debug_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_x = int(center_orig_x - text_width / 2)
            text_y = int(center_orig_y + text_height / 2)
            cv2.putText(
                final_result,
                debug_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (220, 220, 220),
                thickness,
                cv2.LINE_AA,
            )

            # Write frame
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
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                f"scale='trunc(iw/2)*2':'trunc(ih/2)*2',format=pix_fmts=yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-movflags",
                "+faststart",
                "-shortest",
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
                    f"FFmpeg error (code {result.returncode}):\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
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
        # Clean up temporary files on error
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
