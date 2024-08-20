from typing import AsyncGenerator, Generator, Optional
from fakts.fakts import get_current_fakts
from koil import unkoil
from livekit import rtc
import cv2
from arkitekt_next import register
from unlok_next.api.schema import acreate_stream, CreateStreamInput, Stream
import asyncio
import janus
import threading
import time
import numpy as np
import xarray as xr

from arkitekt_next import register, startup
from mikro_next.api import (
    Image,
    from_array_like,
    afrom_array_like,
)
from unlok_next.api.schema import acreate_room
from rekuest_next import model
from dataclasses import dataclass
import asyncio
import janus

import asyncio

import numpy as np

from akuire.acquisition import Acquisition
from akuire.compilers.default import compile_events
from akuire.config import SystemConfig
from akuire.engine import AcquisitionEngine
from akuire.events import AcquireTSeriesEvent
from akuire.events.data_event import ImageDataEvent
from akuire.managers.virtual.smlm_microscope import SMLMMicroscope

# Settting up the acquisition engine
# This is based on the akuire enginge but should be using
# the new imswitch xengine instead
engine = AcquisitionEngine(
    system_config=SystemConfig(
        managers=[SMLMMicroscope("smlm_microscope")],
    ),
    compiler=compile_events,
)

# The model decorator is necessary to tell arkitekt that this type is
# consistend of parsable subtypes (int, Images, Streams are all Arkitekt Types)
@model
@dataclass
class State:
    """ State of the microscope
    
    States are used to keep track of the current state of the microscope
    It is communicated back to the user interface to show the current state
    of the microscope. Here we keep track of the current position of the
    microscope and the latest image acquired. As well as the active (WebRTC) stream

    
    """
    position_x: Optional[int] = 0
    position_y: Optional[int] = 0
    position_z: Optional[int] = 0
    latest_image: Optional[Image] = None
    stream: Optional[Stream] = None


# We initialize the global state of the microscope (should read from the micrscope at the start)
global_state = State()



class PubSub:
    """ A simple Publish Subscripe pattern (PubSUB to allow for state updates to be broadcasted to subscribers

    As multiple listeneres (/GUIs might be interested in the state of the microscope), we
    need a way to broadcast the state to all subscribers. 
    """
    def __init__(self):
        self.subscribers = []

    async def asubscribe(self):
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    async def apublish(self, state):
        for subscriber in self.subscribers:
            await subscriber.put(state)

    def publish(self, state):
        # This is using unkoil to bridge between the asyncio and sync world
        return unkoil(self.apublish, state)


state_pubsub = PubSub()


# The __state__ function is a special function that is used to subscribe to the state
# of the app. This is used to keep the GUI up to date with the current state of the microscope
# when changes are made to the state by the mciroscope. __state__ functions are currently
# autosubscribted to when we open the GUI on the apps page (not sure if this is good)
@register(name="State")
async def __state__() -> AsyncGenerator[State, None]:
    # Sending the current state to the subscriber
    yield global_state
    print("Subscribing to state")
    try:
        while True:
            # Waiting for the next state update
            queue = await state_pubsub.asubscribe()
            state = await queue.get()
            yield state
    except asyncio.CancelledError as e:
        # We are no longer interested in the state
        # TODO: unsubscribe
        raise e

# The move function is used to move the microscope to a specific position
@register(name="Move")
def move(x: int) -> None:
    """ Move the microscope to a specific position
    
    
    This function is used to move the microscope to a specific position
    in the x direction. The position is communicated to the microscope
    and the state is updated to reflect the new position
    """
    global global_state
    global state_queue
    # We would communicate with the microscope here
    # Faking the movement for now
    time.sleep(1)
    global_state.position_x = x
    state_pubsub.publish(global_state)


# This function is used to acquire an image from the microscope at the current
# position. Here we actually are faking the acquisition by returning a random image
# instead of acquiring an image from the microscope but the process is the same
# This image is not ephermal. ie it will be saved to the mikro database, 
@register(name="Acquire")
def acquire(x: int) -> Image:
    """ Acquire an image from the microscope

    This function is used to acquire an image from the microscope
    at the current position. The image is then returned to the user
    """
    global global_state

    x = from_array_like(
        xr.DataArray(np.random.rand(100, 100, 20), dims=["x", "y", "z"]), name="random"
    )

    global_state.latest_image = x

    state_pubsub.publish(global_state)
    return x


# We rescale the image to 2048x2048 and convert it to an RGBA image
WIDTH = 2048
HEIGHT = 2048

# This function is used to convert a BGR image to an RGBA image, and fit it to the
# the livekti video frame format
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


# This function is used to streaming the video from the microscope to the livekit room

async def stream(room: rtc.Room):
    # get token and connect to room - not included
    # publish a track

    # We are using the livekit library to stream the video to the room
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("hue", source)

    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_CAMERA
    # We start the strea and advertise the track (only when the track is subscibed to will the stream start)
    publication = await room.local_participant.publish_track(track, options)
    
    queue = asyncio.Queue()

    async def image_hook(event: ImageDataEvent):
        """ A hook that is called when an image is acquired from the microscope
        
        This is currently an async function that is called when an image is acquired
        (because akuire is async). Depending on the implementation of the microscope
        this could of course also be sync function, however this should then be
        threaded to not block the main thread
        """

        print("Image acquired")
        if event.data is None:
            return

        else:
            # We convert the image to an RGBA frame and put it in the queue
            # For that we offload the conversion to a thread (to not block the event loop)
            frame = await asyncio.to_thread(convert_to_rgba, event.data)
            print(frame)
            # And finally we put the frame in the queue
            await queue.put(frame)


    # The engine has a pubsub mechanism that is used to subscribe to events
    # that are generated by the microscope. Here we subscribe to the image data
    # events that are generated by the microscope
    engine.add_subscriber(image_hook)

    try:

        # We start listening for images that are acquired from the microscope (and
        # are put into the queue by the image_hook function)
        while True:
            # We get the next frame from the queue
            frame = await queue.get()
            if frame is None:
                continue

            # We capture the frame and send it to the livekit room
            source.capture_frame(frame)
            
            # We could adjust the frame rate here (e.g. by sleeping for a certain amount of time)
            # await asyncio.sleep(1 / 30)  # Adjust sleep time for desired frame rate
    except asyncio.CancelledError:

        # When the stream is cancelled we disconnect from the room

        await room.disconnect()

        # Cancellation Errors should be re-raised (to propagate the cancellation)
        raise


# The start function is used to start the microscope stream
# This function is used to and start waiting for images that are acquired
# from the microscope. The images are then streamed to the livekit room
# This function is designed to be run in the background and will not return
# until the stream has ended. This could also be a sync function but because
# the livekit api is async i prefer to keep it async
@register
async def start() -> Stream:
    global global_state


    # Here we use the lok API to create a room and a stream
    # The creating of rooms is handled through the lok API
    # to enable authentication and to ensure that the current user has the correct permissions

    room = await acreate_room(
        title="Microscope Stream", description="The microscope stream"
    )

    # each app can join a room and stream image data to it
    # ptotentiall you can have multiple streams in a room (e.g. multiple cameras)
    # but for now we only have one stream (so one agent)
    # NOTE: The terminology is a bit confusing here, maybe its best to rename the agentId to something else
    the_stream = await acreate_stream(
        input=CreateStreamInput(room=room.id, title="test", agentId="default")
    )

    # We update the global state to reflect that users can watch the stream now
    global_state.stream = the_stream
    await state_pubsub.apublish(global_state)

    print(the_stream)
    
    # fakts is an arkitekt system that is used to dynamically load configuration
    # from the arkitekt server based on where the app  is running (e.g. when running
    # inside the docker container the configuration is different from when running 
    # throught the vpn)
    fakts = get_current_fakts()

    # We get the correct endpoint to connect to the livekit room
    url = await fakts.aget("livekit.http_endpoint")

    # We connect to the livekit room and start streaming the video
    room = rtc.Room()

    # Livekit rooms are connected using a token that is generated by the server (preauthenticated)
    # This token is used to connect to the room and start streaming as the correct user
    await room.connect(url, the_stream.token)

    # We start the stream
    await stream(room)

    # When the stream ends we update the global state to reflect that the stream has ended

    global_state.stream = None
    await state_pubsub.apublish(global_state)

    return the_stream


@register(name="Acquire Times Series")
async def acquire_snap(t_steps: int) -> None:
    """ Acquire a time series of images from the microscope

    This function is used to acquire a time series of images from the microscope
    at the current position. The images are then returned to the user
    """

    # We are using the akuire engine to acquire the time series of images
    # if previously started the engine hooks will be called when an image is acquired
    # and streamed to the livekit room (on top of that the image is then returned to the user) 
    async with engine as e:
        x = Acquisition(events=[AcquireTSeriesEvent(t_steps=t_steps)])

        collected_frames = []

        async for i in e.acquire_stream(x):
            print("Acquired frame")
            if isinstance(i, ImageDataEvent):
                collected_frames.append(i.data)

        # We return the images to the user
        await afrom_array_like(
            xr.DataArray(np.array(collected_frames), dims=["t", "x", "y"]),
            name="random",
        )
