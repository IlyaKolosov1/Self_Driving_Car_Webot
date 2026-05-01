"""emitter_light controller."""

from controller import Supervisor

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

red_led = supervisor.getDevice("red")
green_led = supervisor.getDevice("green")
receiver = supervisor.getDevice("receiver")
receiver.enable(timestep)

is_green = False

while supervisor.step(timestep) != -1:
    got_green_message = False
    while receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()
        if message == "green":
            got_green_message = True

    is_green = got_green_message
    red_led.set(0 if is_green else 1)
    green_led.set(1 if is_green else 0)

