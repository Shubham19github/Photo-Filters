from App.app import FilterApp
from imutils.video import VideoStream
import time

vs = VideoStream(0).start()
time.sleep(1.0)

sa = FilterApp(vs)
sa.root.mainloop()