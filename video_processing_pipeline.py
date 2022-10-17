import logging
import argparse
from multiprocessing import Pool as ProcessPool, cpu_count
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Numeric

import numpy as np
from os import listdir
from os.path import isfile, join
import shutil
import os


import cv2

RELATIVE_INPUT_VIDEOS_FOULDER_PATH = "./in"
RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH = "./out"
DATABASE_FILE_NAME = "video_frames_database.db"
SQLITE_DRIVER_NAME = "sqlite:///"
BATCH_SIZE_TO_DB = 100
VIDEO_FILES_EXTENSION = '.mp4'
TARGET_WIDTH = 30
TARGET_HEIGHT = 30


Base = declarative_base()


class FrameData(Base):
    __tablename__ = "video_frames"
    video_name = Column(String, primary_key=True)
    frame_number = Column(Integer, primary_key=True)
    frame_timestamp = Column(Numeric)
    frame_string = Column(String)


class ToDataBaseBatchSender(list):

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def append(self, item):
        super().append(item)
        if len(self) > BATCH_SIZE_TO_DB:
            self.batch_load_to_db()

    def batch_load_to_db(self):
        logger = logging.getLogger("log")
        logger.debug("Start loading batch to database")
        with Session(self.engine) as session:
            session.add_all(self)
            session.commit()
        logger.debug("Finished loading batch to database")
        self.clear()


def process_multiple_videos(filenames, logger_is_debug):
    logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
    logger = logging.getLogger("log")
    if logger_is_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    engine = create_engine(SQLITE_DRIVER_NAME + DATABASE_FILE_NAME)
    engine.connect()
    batchSender = ToDataBaseBatchSender(engine)
    cap_dict = {}
    for filename in filenames:
        cap = cv2.VideoCapture(
            RELATIVE_INPUT_VIDEOS_FOULDER_PATH + '/' + filename)
        cap_dict[filename] = cap
    for filename in cap_dict.keys():
        logger.info("Start processing video file {}".format(filename))
        cap = cap_dict[filename]
        frame_ind = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(
                batchSender, frame, frame_ind, filename, cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_ind += 1
        logger.info("Finish processing video file {}".format(filename))
        logger.info("Moving video file {} to out foulder".format(filename))
        shutil.move(RELATIVE_INPUT_VIDEOS_FOULDER_PATH + '/' +
                    filename, RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH)
        logger.info(
            "Finished moving video file {} to out foulder".format(filename))
        logger.debug("Sending batch of data to database")
        batchSender.batch_load_to_db()
        logger.debug("Finished sending batch of data to database")
    engine.dispose()


def process_frame(batch_sender, frame, frame_ind, filename, timestamp):
    logger = logging.getLogger("log")
    logger.debug("Processing frame {}".format(
        str(frame_ind) + " of video file " + filename))
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT),
                       interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_OTSU)
    frame = np.array(frame)
    frame //= np.amax(frame)
    frame = "".join(frame.reshape(TARGET_WIDTH * TARGET_HEIGHT).astype(str))
    frameToDB = FrameData(video_name=filename,
                          frame_number=frame_ind,
                          frame_timestamp=timestamp,
                          frame_string=frame
                          )
    batch_sender.append(frameToDB)
    logger.debug("Finished processing frame {}".format(
        str(frame_ind) + " of video file " + filename))


def get_all_mp4_files_in_foulder(path):
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    mp4_files = list(
        filter(lambda x: x[-4:] == VIDEO_FILES_EXTENSION, all_files))
    return mp4_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes_number", default=cpu_count(), type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    processes_number = args.processes_number
    logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
    logger = logging.getLogger("log")
    if args.debug == True:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Started")
    filenames = get_all_mp4_files_in_foulder(
        RELATIVE_INPUT_VIDEOS_FOULDER_PATH)
    try:
        os.mkdir(RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH)
    except OSError as error:
        print(error)
    filenames_splitted = list(np.array_split(
        np.array(filenames), processes_number))

    engine = create_engine(SQLITE_DRIVER_NAME + DATABASE_FILE_NAME)
    engine.connect()
    Base.metadata.create_all(engine)
    engine.dispose()
    filenames_splitted_plus_logger_level = []
    for file_names_group in filenames_splitted:
        filenames_splitted_plus_logger_level.append(
            (file_names_group, args.debug))
    with ProcessPool(processes_number) as pool:
        pool.starmap(process_multiple_videos,
                     filenames_splitted_plus_logger_level)
    logger.info("Completed")


if __name__ == '__main__':
    main()
