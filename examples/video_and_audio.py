import asyncio
import base64
import logging
import multiprocessing as mp
import threading
from queue import Empty, Queue
from typing import Any

import av
import click
import cv2
import numpy as np
import numpy.typing as npt
import sounddevice as sd
from aiortsp.rtsp.reader import RTSPReader
from pupil_labs.realtime_api.streaming.nal_unit import extract_payload_from_nal_unit
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

BGRBuffer = npt.NDArray[np.uint8]


def video_decoder_process(
    raw_packet_queue: mp.Queue,
    decoded_frame_queue: mp.Queue,
    video_media: dict[str, Any],
):
    """Dedicated decoding process.

    Pulls raw packets, decodes them, and puts the decoded frames into a queue for the
    renderer.
    """
    logging.info("Video decoder process started")
    codec = av.CodecContext.create("h264", "r")
    try:
        fmtp = video_media.get("attributes", {}).get("fmtp", {})
        sprop = fmtp["sprop-parameter-sets"]
        params = [base64.b64decode(p) for p in sprop.split(",")]
        for param in params:
            codec.parse(extract_payload_from_nal_unit(param))
    except Exception as e:
        logging.exception(f"Error setting up H.264 decoder: {e}")
        return

    try:
        while True:
            raw_data = raw_packet_queue.get()
            if raw_data is None:
                break

            parsed_data = extract_payload_from_nal_unit(raw_data)
            for packet in codec.parse(parsed_data):
                for frame in codec.decode(packet):
                    decoded_frame_queue.put(frame.to_ndarray(format="bgr24"))
    finally:
        logging.info("Video decoder process finished.")
        decoded_frame_queue.put(None)


def audio_decoder_thread_target(
    raw_audio_queue: mp.Queue,
    decoded_audio_queue: Queue,
    audio_media: dict[str, Any],
    stop_event: threading.Event,
):
    """Runs in a thread inside the audio process.

    Pulls raw audio from network, decodes it, and puts it in an internal queue
    as NumPy arrays.
    """
    logging.info("Audio decoder thread started.")
    codec = av.CodecContext.create("aac", "r")
    resampler = None
    try:
        fmtp = audio_media.get("attributes", {}).get("fmtp", {})
        if "config" in fmtp:
            import binascii

            extradata = binascii.unhexlify(fmtp["config"])
            codec.extradata = extradata
    except Exception as e:
        logging.exception(f"Error setting up AAC decoder: {e}")
        return

    while not stop_event.is_set():
        try:
            pkt_data = raw_audio_queue.get(timeout=0.1)
            if pkt_data is None:
                break

            try:
                au_headers_length_bits = int.from_bytes(pkt_data[:2], "big")
                au_headers_length_bytes = (au_headers_length_bits + 7) // 8
                payload_start = 2 + au_headers_length_bytes
                au_header_val = int.from_bytes(pkt_data[2:4], "big")
                frame_size = au_header_val >> 3
                aac_frame_data = pkt_data[payload_start : payload_start + frame_size]

                for frame in codec.decode(av.Packet(aac_frame_data)):
                    if not resampler:
                        resampler = av.AudioResampler(
                            format="s16",
                            layout=frame.layout.name,
                            rate=frame.sample_rate,
                        )
                    for resampled_frame in resampler.resample(frame):
                        decoded_audio_queue.put(resampled_frame.to_ndarray())
            except Exception as e:
                logging.warning(f"Audio decoding error: {e}")
        except Empty:
            continue
    decoded_audio_queue.put(None)
    logging.info("Audio decoder thread finished.")


def audio_process_target(raw_audio_queue: mp.Queue, audio_media: dict[str, Any]):
    """Dedicated audio process with immediate playback (no priming)."""
    logging.info("Audio consumer process started")
    decoded_audio_queue = Queue(maxsize=30)
    stop_event = threading.Event()

    decoder_thread = threading.Thread(
        target=audio_decoder_thread_target,
        args=(raw_audio_queue, decoded_audio_queue, audio_media, stop_event),
        name="AudioDecoderThread",
    )
    decoder_thread.start()

    audio_buffer = np.array([], dtype=np.int16)

    def audio_callback(
        outdata: npt.NDArray[np.int16],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ):
        """The callback function for the audio stream."""
        nonlocal audio_buffer
        if status:
            logging.warning(f"Audio stream status: {status}")

        while len(audio_buffer) < frames:
            try:
                data = decoded_audio_queue.get_nowait()
                if data is None:
                    raise sd.CallbackStop("End of stream received.")
                audio_buffer = np.concatenate((audio_buffer, data.flatten()))
            except Empty:
                logging.debug("Audio buffer underrun: filling with silence.")
                break

        frames_to_play = audio_buffer[:frames]
        audio_buffer = audio_buffer[frames:]

        outdata[: len(frames_to_play), 0] = frames_to_play
        if len(frames_to_play) < frames:
            outdata[len(frames_to_play) :, 0] = 0

    try:
        sample_rate = (
            audio_media.get("attributes", {}).get("rtpmap", {}).get("clockRate", 8000)
        )
        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=audio_callback
        )
        with stream:
            logging.info("Audio stream started immediately.")
            stop_event.wait()
            logging.info("Stop signal received, preparing to close audio stream.")

    except Exception as e:
        logging.exception(f"An error occurred in the audio process: {e}")
    finally:
        logging.info("Initiating shutdown of audio process.")
        stop_event.set()
        decoder_thread.join(timeout=2.0)
        if decoder_thread.is_alive():
            logging.warning("Decoder thread did not shut down cleanly.")
        logging.info("Audio consumer process finished.")


async def read_stream_to_mp_queue(reader: RTSPReader, queue: mp.Queue, media_type: str):
    """Reads packets from a stream and puts them into a multiprocessing queue."""
    logging.info(f"Packet reader task started for {media_type} stream.")
    try:
        async for pkt in reader.iter_packets():
            queue.put(pkt.data)
    finally:
        logging.info(f"Packet reader task for {media_type} finished.")
        queue.put(None)


def network_process_target(
    url: str, raw_video_queue: mp.Queue, raw_audio_queue: mp.Queue
):
    """Target for the background process. Handles all networking."""
    logging.info("Network Process Started")

    async def main_async():
        try:
            async with (
                RTSPReader(url, media_type="video", timeout=15) as video_reader,
                RTSPReader(url, media_type="audio", timeout=15) as audio_reader,
            ):
                while not (video_reader.session and video_reader.session.sdp):
                    await asyncio.sleep(0.1)
                while not (audio_reader.session and audio_reader.session.sdp):
                    await asyncio.sleep(0.1)

                logging.info("RTSP sessions ready.")

                video_reader_task = asyncio.create_task(
                    read_stream_to_mp_queue(video_reader, raw_video_queue, "video")
                )
                audio_reader_task = asyncio.create_task(
                    read_stream_to_mp_queue(audio_reader, raw_audio_queue, "audio")
                )

                await asyncio.gather(video_reader_task, audio_reader_task)

        except Exception as e:
            logging.exception(f"Error in network process: {e}")

    asyncio.run(main_async())


@click.command()
@click.argument(
    "url", type=str, default="rtsp://192.168.20.165:8086/?camera=world&audioenable=on"
)
def main_cli(url: str):
    """Main entry point: sets up the GUI and the background processes."""
    raw_video_queue = mp.Queue(maxsize=30)
    raw_audio_queue = mp.Queue(maxsize=30)
    decoded_frame_queue = mp.Queue(maxsize=10)
    procs = []

    try:
        # Get media metadata before starting processes
        logging.info("Performing initial connection to get media info...")

        async def get_media_info():
            async with (
                RTSPReader(url, media_type="video", timeout=15) as v_reader,
                RTSPReader(url, media_type="audio", timeout=15) as a_reader,
            ):
                while not (v_reader.session and v_reader.session.sdp):
                    await asyncio.sleep(0.1)
                while not (a_reader.session and a_reader.session.sdp):
                    await asyncio.sleep(0.1)
                return (
                    v_reader.session.sdp.get_media("video"),
                    a_reader.session.sdp.get_media("audio"),
                )

        video_media, audio_media = asyncio.run(get_media_info())
        if not video_media or not audio_media:
            raise RuntimeError("Could not get media information.")
        logging.info("Media info obtained.")

        # Start the background processes
        network_proc = mp.Process(
            target=network_process_target, args=(url, raw_video_queue, raw_audio_queue)
        )
        decoder_proc = mp.Process(
            target=video_decoder_process,
            args=(raw_video_queue, decoded_frame_queue, video_media),
        )
        audio_proc = mp.Process(
            target=audio_process_target, args=(raw_audio_queue, audio_media)
        )
        procs = [network_proc, decoder_proc, audio_proc]
        for p in procs:
            p.start()

        # The main thread is dedicated to the GUI rendering loop
        logging.info("Main thread starting video rendering loop.")
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
        while True:
            try:
                frame = decoded_frame_queue.get(timeout=1)
                if frame is None:
                    break
                cv2.imshow("Video", frame)
            except Empty:
                if not decoder_proc.is_alive():
                    logging.info("Decoder process has terminated.")
                    break
                continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        logging.info("Exiting main thread.")
        # Terminate processes to ensure a clean exit
        for p in procs:
            if p and p.is_alive():
                p.terminate()
            if p:
                p.join(timeout=2)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main_cli()
