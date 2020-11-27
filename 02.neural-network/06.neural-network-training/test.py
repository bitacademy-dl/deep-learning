import datetime

now = datetime.datetime.now()
dstr = f'{now:%Y%m%d%H%M%S}'
print(dstr)
