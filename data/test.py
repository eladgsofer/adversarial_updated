

def turn_on_off(on_off):
    pass

def get_temp():
    return 26

ac_turned = False
from time import sleep
desired_temp = 24
margin = 3

while True:
    curr_temp = get_temp()
    if curr_temp > desired_temp:
        if not ac_turned:
            turn_on_off("on")
            ac_turned = True
    else:
        while abs(get_temp() - desired_temp) < margin:
            sleep(1)

        if ac_turned:
            turn_on_off("off")
            ac_turned = False

    sleep(3)


