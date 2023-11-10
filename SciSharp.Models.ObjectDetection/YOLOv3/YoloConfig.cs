﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SciSharp.Models.ObjectDetection
{
    public class YoloConfig
    {
        public YoloModelConfig YOLO;
        public TrainConfig TRAIN;
        public TestConfig TEST;

        public YoloConfig(string root)
        {
            YOLO = new YoloModelConfig(root);
            TRAIN = new TrainConfig(root);
            TEST = new TestConfig(root);
        }

        public class YoloModelConfig
        {
            string _root;

            public string CLASSES;
            public string ANCHORS;
            public float MOVING_AVE_DECAY = 0.9995f;
            public int[] STRIDES = new int[] { 8, 16, 32 };
            public int ANCHOR_PER_SCALE = 3;
            public float IOU_LOSS_THRESH = 0.5f;
            public string UPSAMPLE_METHOD = "resize";
            public string ORIGINAL_WEIGHT;
            public string DEMO_WEIGHT;

            public YoloModelConfig(string root)
            {
                _root = root;
                CLASSES = Path.Combine(_root, "data", "classes", "yymnist.names");
                ANCHORS = Path.Combine(_root, "data", "anchors", "basline_anchors.txt");
                ORIGINAL_WEIGHT = Path.Combine(_root, "checkpoint", "yolov3_coco.ckpt");
                DEMO_WEIGHT = Path.Combine(_root, "checkpoint", "yolov3_coco_demo.ckpt");
            }
        }

        public class TrainConfig
        {
            string _root;

            public int BATCH_SIZE = 4;
            // new int[] { 320, 352, 384, 416, 448, 480, 512, 544, 576, 608 };
            public int[] INPUT_SIZE = new int[] { 416, 416, 3 }; 
            public bool DATA_AUG = true;
            public float LEARN_RATE_INIT = 1e-3f;
            public float LEARN_RATE_END = 1e-6f;
            public int WARMUP_EPOCHS = 2;
            public int EPOCHS = 30;
            public string INITIAL_WEIGHT;
            public string ANNOT_PATH;

            public TrainConfig(string root)
            {
                _root = root;
                INITIAL_WEIGHT = Path.Combine(_root, "checkpoint", "yolov3_coco_demo.ckpt");
                ANNOT_PATH = Path.Combine(_root, "data", "dataset", "yymnist_train.txt");
            }
        }

        public class TestConfig
        {
            string _root;

            public int BATCH_SIZE = 3;
            public int[] INPUT_SIZE = new int[] { 416, 416, 3 };
            public bool DATA_AUG = false;
            public bool WRITE_IMAGE = true;
            public string DECTECTED_IMAGE_PATH;
            public string WEIGHT_FILE;
            public bool WRITE_IMAGE_SHOW_LABEL = true;
            public bool SHOW_LABEL = true;
            public float SCORE_THRESHOLD = 0.3f;
            public float IOU_THRESHOLD = 0.45f;
            public string ANNOT_PATH;
            
            public TestConfig(string root)
            {
                _root = Path.GetFullPath(root);
                ANNOT_PATH = Path.Combine(_root, "data", "dataset", "yymnist_test.txt");
                DECTECTED_IMAGE_PATH = Path.Combine(_root, "data", "detection");
                Directory.CreateDirectory(DECTECTED_IMAGE_PATH);
                WEIGHT_FILE = Path.Combine(_root, "checkpoint", "yolov3_test_loss=9.2099.ckpt-5");
            }
        }
    }
}
