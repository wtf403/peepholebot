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
    if message.from_user.id in processing_users:
        await message.answer(
            "Another video note is currently being processed. Please wait."
        )
        return

    processing_users.add(message.from_user.id)
    processing = await message.answer(
        "⏳ Processing your video note...", reply_markup=ReplyKeyboardRemove()
    )

    try:
        video_note = message.video_note
        file = await message.bot.get_file(video_note.file_id)
        file_path = file.file_path

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_input:
            await message.bot.download_file(file_path, destination=temp_input.name)
            input_path = temp_input.name

        output_path = await process_video_with_peephole_effect(input_path)

        await processing.delete()
        await message.reply_video_note(
            FSInputFile(output_path),
            reply_to_message_id=processing.message_id,
            reply_markup=KB,
        )

        os.unlink(input_path)
        os.unlink(output_path)

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        await message.answer(
            "Sorry, there was an error processing your video",
        )
    finally:
        processing_users.remove(message.from_user.id)


async def process_video_with_peephole_effect(input_path):
    output_path = f"{input_path}_processed.mp4"
    temp_output_path = f"{input_path}_temp.mp4"

    def _process_video():
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Create circle mask and vignette mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width // 2, height // 2), width // 2, 255, -1)

        # Create a softer edge mask for vignette effect
        vignette_mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(vignette_mask, (width // 2, height // 2), width // 2, 1.0, -1)
        # Blur the edges to create gradient
        vignette_mask = cv2.GaussianBlur(
            vignette_mask, (width // 10 + 1, width // 10 + 1), 0
        )

        # Create maps for the lens effect
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        center_x = width // 2
        center_y = height // 2
        center_radius = width * 0.4  # 40% of the input size will be preserved exactly

        # Pre-fill the maps with identity transform (no effect)
        for y in range(height):
            for x in range(width):
                map_x[y, x] = x
                map_y[y, x] = y

        # Apply lens effect only where needed
        for y in range(height):
            for x in range(width):
                # Calculate distance from center (in pixels)
                dx = x - center_x
                dy = y - center_y
                r_pixels = np.sqrt(dx * dx + dy * dy)

                # Skip center region - we want to preserve it exactly
                if r_pixels <= center_radius:
                    continue

                # Calculate normalized radius (0-1 scale where 1 is edge of circle)
                r = r_pixels / (width / 2)

                if r >= 1.0:
                    # Outside the circle - no need to process
                    continue

                # Calculate angle - we'll preserve this exactly
                theta = np.arctan2(dy, dx)

                # Apply lens effect - edges only
                # Calculate how far we are into the edge region
                edge_t = (r_pixels - center_radius) / (width / 2 - center_radius)
                edge_t = min(1.0, edge_t)  # Cap at 1.0

                # More subtle lens effect - reduced strength
                lens_strength = 0.3 * edge_t**3

                # New radius after lens effect (compression toward center)
                r_new = r_pixels * (1.0 - lens_strength * edge_t)

                # Convert back to cartesian coordinates while preserving angle
                new_x = center_x + r_new * np.cos(theta)
                new_y = center_y + r_new * np.sin(theta)

                # Here we're doing the inverse mapping - which pixel in source maps to destination
                map_x[y, x] = new_x
                map_y[y, x] = new_y

        # Create a background for areas outside the circle
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background.fill(20)

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the mapping (lens effect)
            distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

            # Apply circular mask and vignette
            masked_frame = cv2.bitwise_and(distorted, distorted, mask=mask)
            vignette = masked_frame.copy()
            for c in range(3):
                vignette[:, :, c] = vignette[:, :, c] * vignette_mask

            # Combine with background
            result_frame = background.copy()
            result_frame = cv2.add(result_frame, vignette)
            out.write(result_frame)

        cap.release()
        out.release()

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_output_path,
                    "-i",
                    input_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "15",  # Better quality (lower value)
                    "-preset",
                    "medium",
                    "-tune",
                    "film",  # Better for video content
                    "-x264-params",
                    "keyint=30:min-keyint=15:no-scenecut=1",  # Better keyframe interval
                    "-pix_fmt",
                    "yuv420p",  # Ensure proper pixel format
                    "-c:a",
                    "aac",  # Use standard audio codec
                    "-b:a",
                    "192k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-movflags",
                    "+faststart",  # Enable streaming
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            os.unlink(temp_output_path)
        except Exception as e:
            logging.error(f"FFmpeg merging error: {e}")
            os.rename(temp_output_path, output_path)

    await asyncio.to_thread(_process_video)

    return output_path


async def main():
    bot = Bot(token=TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
