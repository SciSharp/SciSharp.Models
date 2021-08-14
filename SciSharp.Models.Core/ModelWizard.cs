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

        public IModelTask AddTask(IModelTask task)
        {
            _context.Task = task;
            return task;
        }
    }
}
