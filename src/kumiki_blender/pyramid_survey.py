"""Historic Khufu course elevations for a measured-height pyramid study.

Source: W. M. F. Petrie, *The Pyramids and Temples of Gizeh* (1883), Plate 8,
N.E. course top elevations in inches, as corrected/transcribed by Ronald
Birdsall (2003), https://www.ronaldbirdsall.com/gizeh/errata/levels.html .
These are historical measured LEVELS, not a contemporary 3-D survey of each
block's footprint, and not a 3-D reconstruction of individual stone damage.
"""

from __future__ import annotations

from .architecture_geometry import GeometryError


KHUFU_ORIGINAL_BASE_M = 230.4  # Approximate full base; not taken from course table.
KHUFU_ORIGINAL_HEIGHT_M = 146.5  # Original apex, now missing.
PETRIE_INCH_M = 0.0254
PETRIE_LAST_PRESERVED_COURSE = 201  # Petrie notes courses 202–203 are now gone.
PETRIE_LEVELS_SOURCE_URL = "https://www.ronaldbirdsall.com/gizeh/errata/levels.html"

# One-based course number n -> PETRIE_NE_COURSE_TOP_IN[n - 1].
PETRIE_NE_COURSE_TOP_IN: tuple[float, ...] = (
    58.6, 107.7, 155.9, 199.9, 240.1, 278.1, 317.3, 355.1, 391.5, 427.5,
    461.2, 490.9, 520.7, 550.1, 579.4, 608.3, 636.3, 667.4, 705.5, 729.0,
    752.8, 787.3, 820.3, 852.7, 885.0, 915.8, 945.5, 974.7, 1002.8, 1031.0,
    1058.8, 1085.1, 1111.4, 1137.6, 1187.4, 1228.5, 1266.7, 1303.0, 1336.9, 1368.8,
    1399.7, 1427.7, 1461.0, 1502.1, 1540.0, 1567.3, 1602.9, 1638.4, 1669.6, 1697.6,
    1724.2, 1750.5, 1776.1, 1801.8, 1827.1, 1851.9, 1874.4, 1903.0, 1931.7, 1959.8,
    1986.0, 2010.4, 2036.4, 2062.0, 2088.2, 2111.9, 2146.9, 2177.5, 2209.9, 2237.7,
    2266.5, 2292.3, 2317.6, 2348.0, 2377.0, 2400.6, 2426.1, 2449.8, 2472.9, 2497.0,
    2520.2, 2543.5, 2566.1, 2593.9, 2616.8, 2643.1, 2665.0, 2688.1, 2710.8, 2749.3,
    2783.6, 2816.7, 2846.1, 2872.2, 2897.8, 2920.8, 2946.8, 2984.3, 3022.6, 3058.2,
    3091.7, 3119.1, 3148.4, 3174.7, 3201.1, 3226.1, 3250.9, 3280.2, 3306.8, 3330.0,
    3352.9, 3377.0, 3400.2, 3423.2, 3445.3, 3472.4, 3494.8, 3530.4, 3562.2, 3592.0,
    3621.1, 3647.1, 3673.5, 3697.8, 3722.6, 3745.6, 3768.2, 3791.0, 3813.1, 3839.7,
    3864.6, 3887.8, 3909.3, 3932.1, 3953.3, 3977.0, 3999.3, 4025.0, 4050.5, 4072.2,
    4093.2, 4116.3, 4137.2, 4168.3, 4192.5, 4216.4, 4238.5, 4260.4, 4282.1, 4308.7,
    4333.1, 4356.1, 4378.3, 4400.1, 4420.6, 4441.9, 4463.1, 4484.7, 4505.3, 4526.8,
    4547.5, 4571.3, 4593.4, 4619.2, 4640.6, 4661.8, 4682.4, 4702.4, 4722.9, 4743.7,
    4763.4, 4784.6, 4805.4, 4825.1, 4845.4, 4865.9, 4886.2, 4906.3, 4926.9, 4953.4,
    4978.1, 5001.4, 5023.9, 5046.1, 5066.8, 5087.9, 5107.6, 5128.5, 5149.0, 5169.4,
    5190.6, 5210.7, 5231.3, 5252.0, 5271.8, 5295.7, 5319.2, 5341.3, 5362.9, 5385.7,
    5407.9, 5430.4, 5451.8,
)


def petrie_course_tops_m(*, last_course: int = PETRIE_LAST_PRESERVED_COURSE) -> tuple[float, ...]:
    """Measured NE corner course tops, from the base in meters, not evenly spaced.

    The last two recorded remnants (202 and 203) were subsequently destroyed.
    We default to 201 to avoid inventing a new apex, but do not claim every
    course or stone is fully preserved on the contemporary monument.
    """
    if type(last_course) is not int or not 1 <= last_course <= 203:
        raise GeometryError("last_course must be an integer from 1 to 203")
    tops = tuple(v * PETRIE_INCH_M for v in PETRIE_NE_COURSE_TOP_IN[:last_course])
    if any(a >= b for a, b in zip(tops, tops[1:])):
        raise GeometryError("Petrie course levels must be strictly increasing")
    return tops
