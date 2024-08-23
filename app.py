from typing import AsyncGenerator, Generator, Optional

from pydantic import BaseModel
from fakts.fakts import get_current_fakts
from koil import unkoil
from livekit import rtc
import cv2
from arkitekt_next import background, easy, register
from rekuest_next.agents.context import context
from rekuest_next.state.state import state
from unlok_next.api.schema import acreate_stream, CreateStreamInput, Stream
import asyncio
import janus
import threading

import numpy as np
import xarray as xr

from arkitekt_next import register, startup
from mikro_next.api import (
    Image,
    from_array_like,
)
from unlok_next.api.schema import acreate_room
from rekuest_next import model
from dataclasses import dataclass
import asyncio
import janus
from rekuest_next.api.schema import PanelKind, UIChildInput, UIChildKind, UITreeInput, acreate_dashboard, acreate_panel, create_dashboard, CreateDashboardInput, create_panel, CreatePanelInput
import asyncio

import numpy as np

from akuire.acquisition import Acquisition
from akuire.compilers.default import compile_events
from akuire.config import SystemConfig
from akuire.engine import AcquisitionEngine
from akuire.events import AcquireTSeriesEvent
from akuire.events.data_event import ImageDataEvent
from akuire.managers.virtual.smlm_microscope import SMLMMicroscope

# Set the threshold for detecting a scream
WIDTH = 2048
HEIGHT = 2048


engine = AcquisitionEngine(
    system_config=SystemConfig(
        managers=[SMLMMicroscope("smlm_microscope")],
    ),
    compiler=compile_events,
)

@state("default")
class AppState(BaseModel):
    latest_image: Optional[Image] = None
    position_x: int = 0
    position_y: int = 0
    stream: Optional[Stream] = None

@context("db")
class DBContext(BaseModel):
    db_connection: str



@register(name="Move")
def move(x: int, state: AppState) -> None:
    state.position_x = x


@register(name="Acquire")
def acquire(x: int, state: AppState) -> Image:

    x = from_array_like(
        xr.DataArray(np.random.rand(100, 100, 20), dims=["x", "y", "z"]), name="random"
    )

    state.latest_image = x

    return x



def convert_to_rgba(frame: np.ndarray) -> rtc.VideoFrame:
    """Convert a BGR frame to an RGBA frame."""

    frame = frame[:, :]
    # Resize frame if necessary

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # rescale to 0-255
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    frame = frame.astype(np.uint8)

    # Convert BGR frame to RGBA format
    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)

    # Create a VideoFrame and capture it
    frame_data = rgba_frame.tobytes()
    frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, frame_data)
    return frame


async def stream(room: rtc.Room):
    # get token and connect to room - not included
    # publish a track
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("hue", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_CAMERA
    publication = await room.local_participant.publish_track(track, options)
    event = threading.Event()
    queue = asyncio.Queue()

    async def image_hook(event: ImageDataEvent):
        print("Image acquired")
        if event.data is None:
            return

        else:
            frame = await asyncio.to_thread(convert_to_rgba, event.data)
            print(frame)
            await queue.put(frame)

    engine.add_subscriber(image_hook)

    try:
        while True:
            frame = await queue.get()
            if frame is None:
                continue
            source.capture_frame(frame)

            # await asyncio.sleep(1 / 30)  # Adjust sleep time for desired frame rate
    except asyncio.CancelledError:
        event.set()

        await room.disconnect()

        raise


@startup
async def start(instance_id) -> AppState:

    print("Starting", instance_id)

    z = await acreate_panel(
            input=CreatePanelInput(
                kind=PanelKind.STATE,
                stateKey="default:stream",
                instanceId=instance_id,
            )
        )

    y = await acreate_panel(
        input=CreatePanelInput(
            kind=PanelKind.ASSIGN,
            interface="acquire_snap",
            instanceId=instance_id,
        )
    )


    t = await acreate_dashboard(input=CreateDashboardInput(
        name="karlo",
        panels=[z,y]
    ))


    room = await acreate_room(
        title="Microscope Stream", description="The microscope stream"
    )

    the_stream = await acreate_stream(
        input=CreateStreamInput(room=room.id, title="test", agentId=instance_id)
    )

    print(the_stream)

    return AppState(stream=the_stream)


@background
async def stream_background(state: AppState):
   
    fakts = get_current_fakts()

    url = await fakts.get("livekit.http_endpoint")
    room = rtc.Room()
    await room.connect(url, (await state.stream.aget()).token)
    await stream(room)



@register(name="Acquire Times Series")
async def acquire_snap(t_steps: int) -> None:
    async with engine as e:
        x = Acquisition(events=[AcquireTSeriesEvent(t_steps=t_steps)])

        collected_frames = []

        async for i in e.acquire_stream(x):
            print("Acquired frame")
            if isinstance(i, ImageDataEvent):
                collected_frames.append(i.data)










