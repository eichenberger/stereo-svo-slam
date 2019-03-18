import io

def set_manual_exposure(hidraw, value):
    MAX_EXPOSURE=300000
    if value >= MAX_EXPOSURE:
        print(f'Exposure must be less than {MAX_EXPOSURE} (is {value})')
        return
    f = io.open(hidraw, 'wb', buffering=0)
    data = bytes([0x78, 0x02, (value >> 24)&0xFF, (value >> 16)&0xFF, (value>>8)&0xFF, value&0xFF])
    f.write(data)
    f.close()

def set_auto_exposure(hidraw):
    set_manual_exposure(hidraw, 1)

