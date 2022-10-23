import argparse
import logging
import queue
import shutil
from multiprocessing import Process, cpu_count
from os import listdir, mkdir
from os.path import isfile, join
from threading import Thread

import cv2
import numpy as np
from sqlalchemy import Column, Integer, Numeric, String, create_engine
from sqlalchemy.orm import Session, declarative_base

RELATIVE_INPUT_VIDEOS_FOULDER_PATH = "./in"
RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH = "./out"
DATABASE_FILE_NAME = "video_frames_database.db"
SQLITE_DRIVER_NAME = "sqlite:///"
BATCH_SIZE_TO_DB = 1000
QUEUE_OF_TASKS_SIZE = 200
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
        logger.info("Start loading batch to database")
        with Session(self.engine) as session:
            session.add_all(self)
            session.commit()
        logger.info("Finished loading batch to database")
        self.clear()


def get_all_mp4_files_in_foulder(path):
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    mp4_files = list(
        filter(lambda x: x[-4:] == VIDEO_FILES_EXTENSION, all_files))
    return mp4_files


def process_videos(filenames, processes_number):
    logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    engine = create_engine(SQLITE_DRIVER_NAME + DATABASE_FILE_NAME)
    engine.connect()
    batch_sender = ToDataBaseBatchSender(engine)
    read_frames_queue = queue.Queue(QUEUE_OF_TASKS_SIZE // processes_number)
    thread = Thread(target=read_frames, args=(read_frames_queue, filenames))
    thread.start()
    while True:
        data = read_frames_queue.get()
        if data is None:
            break
        frame_raw, frame_ind, filename, timestamp = data[0], data[1], data[2], data[3]
        frame = process_frame(frame_raw)
        batch_sender.append(FrameData(video_name=filename,
                                      frame_number=frame_ind,
                                      frame_timestamp=timestamp,
                                      frame_string=frame
                                      ))
    thread.join()
    batch_sender.batch_load_to_db()


def read_frames(read_frames_queue, filenames):
    for filename in filenames:
        logger = logging.getLogger("log")
        logger.info("Start reading video file {}".format(filename))
        cap = cv2.VideoCapture(
            RELATIVE_INPUT_VIDEOS_FOULDER_PATH + '/' + filename)
        frame_ind = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_ind += 1
            if not ret:
                break
            read_frames_queue.put(
                (frame, frame_ind, filename, cap.get(cv2.CAP_PROP_POS_MSEC)))
        logger.info("Finished reading video file {}".format(filename))
        logger.info(
            "Started moving video file {} to out foulder".format(filename))
        shutil.move(RELATIVE_INPUT_VIDEOS_FOULDER_PATH + '/' +
                    filename, RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH)
        logger.info(
            "Finished moving video file {} to out foulder".format(filename))
    read_frames_queue.put(None)


def process_frame(frame):
    frame_processed = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT),
                                 interpolation=cv2.INTER_AREA)
    frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    th, frame_processed = cv2.threshold(
        frame_processed, 128, 255, cv2.THRESH_OTSU)
    frame_processed = np.array(frame_processed)
    frame_processed //= np.amax(frame_processed)
    frame_processed = "".join(frame_processed.reshape(
        TARGET_WIDTH * TARGET_HEIGHT).astype(str))
    return frame_processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes_number", default=cpu_count(), type=int)
    args = parser.parse_args()
    logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    logger.info("Started")
    filenames = get_all_mp4_files_in_foulder(
        RELATIVE_INPUT_VIDEOS_FOULDER_PATH)
    try:
        mkdir(RELATIVE_OUTPUT_VIDEOS_FOULDER_PATH)
    except OSError as error:
        logger.info(error)
    engine = create_engine(SQLITE_DRIVER_NAME + DATABASE_FILE_NAME)
    engine.connect()
    Base.metadata.create_all(engine)
    engine.dispose()
    filenames_chunks = np.array_split(filenames, args.processes_number)
    for i in range(len(filenames_chunks)):
        filenames_chunks[i] = filenames_chunks[i].tolist()
    processes = list()
    for i in range(args.processes_number):
        process = Process(target=process_videos, args=(
            filenames_chunks[i], args.processes_number))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    logger.info("Completed")


if __name__ == '__main__':
    main()
