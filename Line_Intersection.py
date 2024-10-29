from shapely.geometry import LineString, Polygon


class WrongLineIntersection:

    def __init__(self, xyxy, line_zones):
        self.xyxy = xyxy
        self.line = LineString([line_zones[0], line_zones[1]])

    def point_line_intersection_test(self):
        try:
            any_intersecting_or_touching_or_standing = False

            p_x1, p_y1, p_x2, p_y2 = self.xyxy.astype(int)
            person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]

            rect_polygon = Polygon(person_coord)

            # Check if the rectangle intersects with the line
            intersects = self.line.crosses(rect_polygon)

            if intersects:
                any_intersecting_or_touching_or_standing = True

            return any_intersecting_or_touching_or_standing

        except Exception as ex:
            print(ex)
            return False

