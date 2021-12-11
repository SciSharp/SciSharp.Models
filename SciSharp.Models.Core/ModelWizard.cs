using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models
{
    public class ModelWizard
    {
        ModelContext _context;
        public ModelContext Context => _context;
        public ModelWizard()
        {
            _context = new ModelContext();
        }

        public IImageClassificationTask AddImageClassificationTask<T>(TaskOptions options) 
            where T : IImageClassificationTask, new()
        {
            _context.ImageClassificationTask = new T();
            _context.ImageClassificationTask.Config(options);
            return _context.ImageClassificationTask;
        }

        public IObjectDetectionTask AddObjectDetectionTask<T>(TaskOptions options)
            where T : IObjectDetectionTask, new()
        {
            _context.ObjectDetectionTask = new T();
            _context.ObjectDetectionTask.Config(options);
            return _context.ObjectDetectionTask;
        }
    }
}
