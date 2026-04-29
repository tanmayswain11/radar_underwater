def degree_to_direction(angle):
    dirs=["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int(angle/45)%8]