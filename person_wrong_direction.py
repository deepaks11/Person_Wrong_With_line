import datetime
import time
import numpy as np
import cv2
import supervision as sv
import os
from create_folders import Folder
from Line_Intersection import WrongLineIntersection
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PersonWrongDirection:

    def __init__(self, model, intersection_line, wrong_way_cam_path,  image_writer_flag=True):

        self.model = model
        self.draw_line = np.array(intersection_line, dtype=np.int32)
        self.intersection_line = intersection_line
        self.entry_line_start = intersection_line[0]  # Start point of the line
        self.entry_line_end = intersection_line[1]  # End point of the line
        self.detections = None
        self.frame = None
        self.xyxy = None
        self.Object_id = []
        self.tracked_ids = {}
        self.in_id = {}
        self.out_id = {}
        self.entry_zone = 3000  # Pixels above the line for "In"
        self.exit_zone = -3000  # Pixels below the line for "Out"
        self.wrong_way_user_cam_path = wrong_way_cam_path + "/detected_images"
        self.Original_path = wrong_way_cam_path + "/original_images"
        self.image_writer_flag = image_writer_flag
        self.detected_image_image_path = []
        self.original_image_path = []
        self.start_time = time.time()
        self.alarm_list = None
        self.alarm_flag = None
        self.wrong_way_flag = False
        self.writer_flag = False
        self.tracked_info = None
        self.count = None
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        Folder(self.wrong_way_user_cam_path).create()
        Folder(self.Original_path).create()

    def write_images(self, detected_image, original_image):

        if self.image_writer_flag:
            detected_image_image_path = self.wrong_way_user_cam_path + "/wrong_way_person{}.jpeg".format(
                str(datetime.datetime.now().strftime("%d_%m_%Y_TIME_%H_%M_%S")))
            # self.detected_image_image_path.append(detected_image_image_path)
            original_image_path = self.Original_path + "/wrong_way_person{}.jpeg".format(
                str(datetime.datetime.now().strftime("%d_%m_%Y_TIME_%H_%M_%S")))
            # self.original_image_path.append(original_image_path)

            cv2.imwrite(detected_image_image_path, detected_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            cv2.imwrite(original_image_path, original_image, [cv2.IMWRITE_JPEG_QUALITY, 50])

    def check_side(self, bbox_center, line_coord):
        """Check if the center of the bbox is above or below the line.

        """
        try:
            line_start, line_end = line_coord
            return (line_end[0] - line_start[0]) * (bbox_center[1] - line_start[1]) - (line_end[1] - line_start[1]) * (
                        bbox_center[0] - line_start[0])
        except Exception as er:
            print(er)

    def find_direction(self, tracker_id, xyxy, direction):

        try:
            self.count = 0
            if direction == "Right":
                direction = "Forward"
            elif direction == "Left":
                direction = "Backward"
            else:
                pass

            x_min, y_min, x_max, y_max = xyxy
            person_center = np.array([(x_min + x_max)/2, (y_min + y_max)/2])
            # Initialize tracking for new persons
            if tracker_id not in self.tracked_ids:
                # New entry, track this person's state
                self.tracked_ids[tracker_id] = {"state": "none"}
                self.in_id[tracker_id] = {"count": 0}
                self.out_id[tracker_id] = {"count": 0}
            tracked_info = self.tracked_ids[tracker_id]

            state = tracked_info["state"]

            current_side = self.check_side(person_center, (self.entry_line_start, self.entry_line_end))
            intersect = WrongLineIntersection(xyxy, self.intersection_line).point_line_intersection_test()

            if state == "none":
                # Check if the person is crossing the line
                if current_side > self.entry_zone and direction == "Backward":
                    self.tracked_ids[tracker_id]["state"] = "crossing_out"

                elif current_side < self.exit_zone and direction == "Forward":
                    self.tracked_ids[tracker_id]["state"] = "crossing_in"

            elif state == "crossing_in" and self.in_id[tracker_id]["count"] == 0 and direction == "Forward":
                if current_side > self.entry_zone and intersect:

                    self.in_id[tracker_id]["count"] = self.in_id[tracker_id]["count"] + 1
                    self.count += 1
                    self.tracked_ids[tracker_id]["state"] = "none"  # Reset state for new counts
                    self.wrong_way_flag = True
                    self.alarm_list.append(True)
                    cv2.polylines(self.frame, [self.draw_line], True, (0, 0, 255), 2)
                    self.frame = cv2.putText(self.frame, "Wrong Way Detected", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif state == "crossing_out" and self.out_id[tracker_id]["count"] == 0 and direction == "Backward":

                if current_side < self.exit_zone and intersect:

                    self.out_id[tracker_id]["count"] = self.out_id[tracker_id]["count"] + 1
                    self.count += 1
                    self.tracked_ids[tracker_id]["state"] = "none"  # Reset state for new counts
                    self.wrong_way_flag = True
                    self.alarm_list.append(True)
                    cv2.polylines(self.frame, [self.draw_line], True, (0, 0, 255), 2)
                    self.frame = cv2.putText(self.frame, "Wrong Way Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                             (0, 0, 255), 2)

        except Exception as er:
            print(er)

    def predict(self, q_img, direction, org_image):
        try:
            # self.alarm_list = []
            # elastic_list = []
            # writer_flag_list = []
            # self.detected_image_image_path = []
            # self.original_image_path = []
            self.wrong_way_flag = False
            # self.alarm_flag = False
            # self.writer_flag = False
            self.frame = q_img.get()

            # Check if a minute has passed to clear the object_id_list
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 60:
                self.Object_id.clear()
                self.start_time = time.time()  # Reset start time after clearing the list

            result = self.model.track(source=self.frame, conf=0.5, classes=0, persist=True, verbose=False)
            result = result[0]

            self.detections = sv.Detections.from_ultralytics(result)

            if self.detections.tracker_id is not None:
                labels = ["{}".format(result.names[class_id]) for xyxy, mask, confidence, class_id, tracker_id, class_name in self.detections]
                self.box_annotator.annotate(scene=self.frame, detections=self.detections)
                self.label_annotator.annotate(scene=self.frame, detections=self.detections, labels=labels)
                for self.xyxy, mask, confidence, class_id, tracker_id, class_name in self.detections:
                    x_min, y_min, x_max, y_max = self.xyxy
                    self.find_direction(tracker_id, self.xyxy, direction)

                    if tracker_id not in self.Object_id and self.wrong_way_flag:
                        cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
                        self.Object_id.append(tracker_id)
                        self.write_images(self.frame, org_image)
                        # elastic_list.append({'x': int(x_min), 'y': int(y_min), 'h': int(x_max), 'w': int(y_max)})
                        # writer_flag_list.append(True)
                    else:
                        continue
                        # writer_flag_list.append(False)

                    # if True in self.alarm_list:
                    #     self.alarm_flag = True
                    # else:
                    #     self.alarm_flag = False
                    #
                    # if True in writer_flag_list:
                    #     self.writer_flag = True
                    # else:
                    #     self.writer_flag = False

                return self.frame
            else:
                return self.frame

        except Exception as er:
            print(er)

