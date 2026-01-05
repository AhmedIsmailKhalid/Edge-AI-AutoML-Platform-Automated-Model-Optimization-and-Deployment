import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { 
  Zap, 
  Upload, 
  ChevronRight, 
  CheckCircle,
  AlertCircle 
} from 'lucide-react';
import { api } from '../api/client';

function CreateExperiment() {
  const navigate = useNavigate();

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    framework: 'pytorch',
    datasetType: 'preset',
    datasetName: 'mnist',
    modelSource: 'pretrained',
    pretrainedModel: '',
    targetDevice: 'raspberry_pi_4',
    optimizationGoal: 'balanced',
    // Optional constraints
    maxAccuracyDrop: '',
    maxSizeMb: '',
    maxLatencyMs: '',
  });

  // File uploads
  const [modelFile, setModelFile] = useState(null);
  const [datasetFile, setDatasetFile] = useState(null);

  // UI state
  const [step, setStep] = useState(1);
  const [errors, setErrors] = useState({});

  // Create experiment mutation
  const createExperimentMutation = useMutation({
    mutationFn: async (data) => {
      try {
        // Step 1: Create experiment
        const expResponse = await api.createExperiment(data);
        const experiment = expResponse.data;
        
        // Step 2: Upload/Load model
        if (formData.modelSource === 'custom' && modelFile) {
          // Custom model upload
          const modelFormData = new FormData();
          modelFormData.append('file', modelFile);
          modelFormData.append('model_source', 'custom');
          await api.uploadModel(experiment.id, modelFormData);
        } else if (formData.modelSource === 'pretrained' && formData.pretrainedModel) {
          // Pretrained model loading
          const modelFormData = new FormData();
          modelFormData.append('model_source', 'pretrained');
          modelFormData.append('pretrained_model_name', formData.pretrainedModel);
          await api.uploadModel(experiment.id, modelFormData);
        }
        
        // Step 3: Upload dataset if custom
        if (formData.datasetType === 'custom' && datasetFile) {
          const datasetFormData = new FormData();
          datasetFormData.append('file', datasetFile);
          datasetFormData.append('dataset_name', datasetFile.name.replace('.zip', ''));
          await api.uploadDataset(experiment.id, datasetFormData);
        }
        
        // Step 4: Start optimization
        await api.startOptimization(experiment.id);
        
        return experiment;
      } catch (error) {
        console.error('Error in experiment creation:', error);
        throw error;
      }
    },
    onSuccess: (experiment) => {
      navigate(`/experiment/${experiment.id}/progress`);
    },
    onError: (error) => {
      console.error('Mutation error:', error);
      setErrors({ 
        submit: error.response?.data?.detail || error.message || 'Failed to create experiment' 
      });
    },
  });

  // Form validation
  const validateStep = (currentStep) => {
    const newErrors = {};

    if (currentStep === 1) {
      if (!formData.name.trim()) {
        newErrors.name = 'Experiment name is required';
      }
      if (!formData.framework) {
        newErrors.framework = 'Framework is required';
      }
    }

    if (currentStep === 2) {
      if (formData.datasetType === 'custom' && !datasetFile) {
        newErrors.dataset = 'Please upload a dataset file';
      }
    }

    if (currentStep === 3) {
      if (formData.modelSource === 'custom' && !modelFile) {
        newErrors.model = 'Please upload a model file';
      }
      if (formData.modelSource === 'pretrained' && !formData.pretrainedModel) {
        newErrors.model = 'Please select a pretrained model';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = (e) => {
    // Prevent any form submission
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (validateStep(step)) {
      console.log(`Moving from step ${step} to ${step + 1}`);
      setStep(step + 1);
    }
  };

  const handleBack = () => {
    setStep(step - 1);
    setErrors({});
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    e.stopPropagation();

    // CRITICAL: Only allow submission on step 5
    if (step !== 5) {
      console.log('⚠️ Form submission blocked - not on step 5');
      return;
    }

    if (!validateStep(step)) return;

    console.log('✅ Submitting experiment...');

    // Build experiment data
    const experimentData = {
      name: formData.name,
      description: `Created via UI - ${formData.framework} - ${formData.datasetName}`,
      framework: formData.framework,
      dataset_type: formData.datasetType,
      dataset_name: formData.datasetType === 'preset' ? formData.datasetName : datasetFile?.name.replace('.zip', ''),
      target_device: formData.targetDevice,
      optimization_goal: formData.optimizationGoal,
    };

    // Add optional constraints
    if (formData.maxAccuracyDrop) {
      experimentData.max_accuracy_drop_percent = parseFloat(formData.maxAccuracyDrop);
    }
    if (formData.maxSizeMb) {
      experimentData.max_size_mb = parseFloat(formData.maxSizeMb);
    }
    if (formData.maxLatencyMs) {
      experimentData.max_latency_ms = parseFloat(formData.maxLatencyMs);
    }

    createExperimentMutation.mutate(experimentData);
  };

  const updateFormData = (field, value) => {
    setFormData({ ...formData, [field]: value });
    setErrors({ ...errors, [field]: undefined });
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Create New Experiment</h1>
        <p className="text-gray-600">
          Configure your model optimization experiment in a few simple steps
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {[1, 2, 3, 4, 5].map((s) => (
            <div key={s} className="flex items-center flex-1">
              <div className="flex items-center space-x-2">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    s < step
                      ? 'bg-green-600 text-white'
                      : s === step
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                  }`}
                >
                  {s < step ? <CheckCircle className="w-6 h-6" /> : s}
                </div>
                <span
                  className={`text-sm font-medium hidden md:block ${
                    s <= step ? 'text-gray-900' : 'text-gray-400'
                  }`}
                >
                  {['Basic', 'Dataset', 'Model', 'Config', 'Review'][s - 1]}
                </span>
              </div>
              {s < 5 && (
                <div
                  className={`h-1 flex-1 mx-2 ${
                    s < step ? 'bg-green-600' : 'bg-gray-200'
                  }`}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Form */}
      <form 
        onSubmit={handleSubmit}
        onKeyDown={(e) => {
          // Prevent Enter key from submitting form unless on final step
          if (e.key === 'Enter' && step !== 5) {
            e.preventDefault();
          }
        }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-8"
      >
        {/* Step 1: Basic Info */}
        {step === 1 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Basic Information</h2>

            {/* Experiment Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Experiment Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => updateFormData('name', e.target.value)}
                placeholder="e.g., mnist_optimization_v1"
                className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent ${
                  errors.name ? 'border-red-500' : 'border-gray-300'
                }`}
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-600 flex items-center space-x-1">
                  <AlertCircle className="w-4 h-4" />
                  <span>{errors.name}</span>
                </p>
              )}
            </div>

            {/* Framework */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Framework *
              </label>
              <div className="grid grid-cols-2 gap-4">
                {['pytorch', 'tensorflow'].map((fw) => (
                  <button
                    key={fw}
                    type="button"
                    onClick={() => updateFormData('framework', fw)}
                    className={`p-4 border-2 rounded-lg text-left transition-all ${
                      formData.framework === fw
                        ? 'border-primary-600 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-semibold text-gray-900 capitalize">{fw}</div>
                    <div className="text-sm text-gray-600 mt-1">
                      {fw === 'pytorch' ? '.pt, .pth files' : '.h5 files'}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Dataset Selection */}
        {step === 2 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Dataset Selection</h2>

            {/* Dataset Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Dataset Type *
              </label>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { value: 'preset', label: 'Preset Dataset', desc: 'MNIST, CIFAR-10, Fashion-MNIST' },
                  { value: 'custom', label: 'Custom Dataset', desc: 'Upload your own dataset' },
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => updateFormData('datasetType', option.value)}
                    className={`p-4 border-2 rounded-lg text-left transition-all ${
                      formData.datasetType === option.value
                        ? 'border-primary-600 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-semibold text-gray-900">{option.label}</div>
                    <div className="text-sm text-gray-600 mt-1">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Preset Dataset Dropdown */}
            {formData.datasetType === 'preset' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Dataset *
                </label>
                <select
                  value={formData.datasetName}
                  onChange={(e) => updateFormData('datasetName', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="mnist">MNIST - Handwritten Digits (28x28 grayscale)</option>
                  <option value="cifar10">CIFAR10 - 10 class Color Images (32x32)</option>
                  <option value="fashionmnist">Fashion MNIST - Fashion Items (28x28 grayscale)</option>
                </select>
              </div>
            )}

            {/* Custom Dataset Upload */}
            {formData.datasetType === 'custom' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Dataset (ZIP) *
                </label>
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const files = e.dataTransfer.files;
                    if (files && files.length > 0) {
                      setDatasetFile(files[0]);
                      setErrors({ ...errors, dataset: undefined });
                    }
                  }}
                  className={`border-2 border-dashed rounded-lg p-8 text-center ${
                    errors.dataset ? 'border-red-500' : 'border-gray-300'
                  }`}
                >
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">
                    {datasetFile ? datasetFile.name : 'Drag and drop or click to upload'}
                  </p>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        setDatasetFile(e.target.files[0]);
                        setErrors({ ...errors, dataset: undefined });
                      }
                    }}
                    className="hidden"
                    id="dataset-upload"
                  />
                  <label
                    htmlFor="dataset-upload"
                    className="inline-block bg-primary-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-primary-700"
                  >
                    Choose File
                  </label>
                  {errors.dataset && (
                    <p className="mt-2 text-sm text-red-600">{errors.dataset}</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Model Selection */}
        {step === 3 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Selection</h2>

            {/* Model Source */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Source *
              </label>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { 
                    value: 'pretrained', 
                    label: 'Pretrained Model', 
                    desc: 'Use our trained models',
                    disabled: formData.datasetType === 'custom'
                  },
                  { 
                    value: 'custom', 
                    label: 'Custom Model', 
                    desc: 'Upload your own model',
                    disabled: false
                  },
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => !option.disabled && updateFormData('modelSource', option.value)}
                    disabled={option.disabled}
                    className={`p-4 border-2 rounded-lg text-left transition-all ${
                      formData.modelSource === option.value
                        ? 'border-primary-600 bg-primary-50'
                        : option.disabled
                        ? 'border-gray-200 bg-gray-100 cursor-not-allowed opacity-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-semibold text-gray-900">{option.label}</div>
                    <div className="text-sm text-gray-600 mt-1">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Pretrained Model Selector */}
            {formData.modelSource === 'pretrained' && formData.datasetType === 'preset' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Pretrained Model *
                </label>
                <select
                  value={formData.pretrainedModel || ''}
                  onChange={(e) => updateFormData('pretrainedModel', e.target.value)}
                  className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent ${
                    errors.model ? 'border-red-500' : 'border-gray-300'
                  }`}
                >
                  <option value="">Choose a model...</option>
                  
                  {formData.datasetName === 'mnist' && (
                    <>
                      <option value="small_mnist_cnn">Small MNIST CNN (Fast, 95% accuracy)</option>
                      <option value="medium_mnist_cnn">Medium MNIST CNN (Balanced, 97% accuracy)</option>
                      <option value="large_mnist_cnn">Large MNIST CNN (Best, 99% accuracy)</option>
                    </>
                  )}
                  
                  {formData.datasetName === 'cifar10' && (
                    <>
                      <option value="small_cifar10_cnn">Small CIFAR-10 CNN (Fast, 70% accuracy)</option>
                      <option value="medium_cifar10_cnn">Medium CIFAR-10 CNN (Balanced, 75% accuracy)</option>
                      <option value="large_cifar10_cnn">Large CIFAR-10 CNN (Best, 80% accuracy)</option>
                    </>
                  )}
                  
                  {formData.datasetName === 'fashionmnist' && (
                    <>
                      <option value="small_fashionmnist_cnn">Small Fashion-MNIST CNN (Fast, 88% accuracy)</option>
                      <option value="large_fashionmnist_cnn">Large Fashion-MNIST CNN (Best, 92% accuracy)</option>
                    </>
                  )}
                </select>
                
                {errors.model && (
                  <p className="mt-2 text-sm text-red-600 flex items-center space-x-1">
                    <AlertCircle className="w-4 h-4" />
                    <span>{errors.model}</span>
                  </p>
                )}
                
                {formData.pretrainedModel && (
                  <p className="mt-2 text-sm text-gray-600">
                    This model is optimized for {formData.datasetName} and trained on {formData.framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}.
                  </p>
                )}
              </div>
            )}

            {/* Custom Model Upload */}
            {formData.modelSource === 'custom' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Model ({formData.framework === 'pytorch' ? '.pt, .pth' : '.h5'}) *
                </label>
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const files = e.dataTransfer.files;
                    if (files && files.length > 0) {
                      setModelFile(files[0]);
                      setErrors({ ...errors, model: undefined });
                    }
                  }}
                  className={`border-2 border-dashed rounded-lg p-8 text-center ${
                    errors.model ? 'border-red-500' : 'border-gray-300'
                  }`}
                >
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">
                    {modelFile ? modelFile.name : 'Drag and drop or click to upload'}
                  </p>
                  <input
                    type="file"
                    accept={formData.framework === 'pytorch' ? '.pt,.pth' : '.h5'}
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        setModelFile(e.target.files[0]);
                        setErrors({ ...errors, model: undefined });
                      }
                    }}
                    className="hidden"
                    id="model-upload"
                  />
                  <label
                    htmlFor="model-upload"
                    className="inline-block bg-primary-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-primary-700"
                  >
                    Choose File
                  </label>
                  {errors.model && (
                    <p className="mt-2 text-sm text-red-600">{errors.model}</p>
                  )}
                </div>
              </div>
            )}

            {/* Pretrained Models Unavailable (Custom Dataset) */}
            {formData.modelSource === 'pretrained' && formData.datasetType === 'custom' && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-900 mb-2">
                      Pretrained Models Unavailable
                    </p>
                    <p className="text-red-800 text-sm mb-4">
                      You have uploaded a custom dataset. Pretrained models are only available for preset datasets (MNIST, CIFAR-10, Fashion-MNIST). Please upload a custom model file.
                    </p>
                    <button
                      type="button"
                      onClick={() => updateFormData('modelSource', 'custom')}
                      className="bg-red-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-red-700"
                    >
                      Switch to Custom Upload
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Step 4: Configuration */}
        {step === 4 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Optimization Configuration</h2>

            {/* Target Device */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Device *
              </label>
              <select
                value={formData.targetDevice}
                onChange={(e) => updateFormData('targetDevice', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="raspberry_pi_3b">Raspberry Pi 3B (1.2GHz, 1GB RAM)</option>
                <option value="raspberry_pi_4">Raspberry Pi 4 (1.5GHz, 4GB RAM)</option>
                <option value="raspberry_pi_5">Raspberry Pi 5 (2.4GHz, 8GB RAM)</option>
                <option value="jetson_nano">NVIDIA Jetson Nano (Maxwell GPU, 4GB RAM)</option>
                <option value="jetson_xavier_nx">NVIDIA Jetson Xavier NX (Volta GPU, 8GB RAM)</option>
                <option value="coral_dev_board">Google Coral Dev Board (Edge TPU, 1GB RAM)</option>
              </select>
            </div>

            {/* Optimization Goal */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Optimization Goal *
              </label>
              <select
                value={formData.optimizationGoal}
                onChange={(e) => updateFormData('optimizationGoal', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="balanced">Balanced - Best trade-off</option>
                <option value="minimize_size">Minimize Size - Smallest model</option>
                <option value="minimize_latency">Maximize Speed - Fastest inference</option>
                <option value="maximize_accuracy">Maximize Accuracy - Preserve accuracy</option>
              </select>
            </div>

            {/* Optional Constraints */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Optional Constraints
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Accuracy Drop (%)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.maxAccuracyDrop}
                    onChange={(e) => updateFormData('maxAccuracyDrop', e.target.value)}
                    placeholder="e.g., 5.0"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Model Size (MB)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.maxSizeMb}
                    onChange={(e) => updateFormData('maxSizeMb', e.target.value)}
                    placeholder="e.g., 10.0"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Latency (ms)
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={formData.maxLatencyMs}
                    onChange={(e) => updateFormData('maxLatencyMs', e.target.value)}
                    placeholder="e.g., 100"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 5: Review */}
        {step === 5 && (
          <div className="space-y-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Review & Submit</h2>

            <div className="bg-gray-50 rounded-lg p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Experiment Name</p>
                  <p className="font-semibold text-gray-900">{formData.name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Framework</p>
                  <p className="font-semibold text-gray-900 capitalize">{formData.framework}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Dataset</p>
                  <p className="font-semibold text-gray-900">
                    {formData.datasetType === 'preset' 
                      ? formData.datasetName 
                      : datasetFile?.name || 'Custom'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Model</p>
                  <p className="font-semibold text-gray-900">
                    {formData.modelSource === 'pretrained' 
                      ? formData.pretrainedModel?.replace(/_/g, ' ') || 'Pretrained'
                      : modelFile?.name || 'Custom'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Target Device</p>
                  <p className="font-semibold text-gray-900 capitalize">
                    {formData.targetDevice.replace(/_/g, ' ')}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Optimization Goal</p>
                  <p className="font-semibold text-gray-900 capitalize">
                    {formData.optimizationGoal.replace(/_/g, ' ')}
                  </p>
                </div>
              </div>

              {(formData.maxAccuracyDrop || formData.maxSizeMb || formData.maxLatencyMs) && (
                <div className="border-t pt-4">
                  <p className="text-sm text-gray-600 mb-2">Constraints</p>
                  <div className="space-y-1">
                    {formData.maxAccuracyDrop && (
                      <p className="text-sm text-gray-900">
                        • Max Accuracy Drop: {formData.maxAccuracyDrop}%
                      </p>
                    )}
                    {formData.maxSizeMb && (
                      <p className="text-sm text-gray-900">
                        • Max Size: {formData.maxSizeMb} MB
                      </p>
                    )}
                    {formData.maxLatencyMs && (
                      <p className="text-sm text-gray-900">
                        • Max Latency: {formData.maxLatencyMs} ms
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>

            {errors.submit && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-800 text-sm flex items-center space-x-2">
                  <AlertCircle className="w-4 h-4" />
                  <span>{errors.submit}</span>
                </p>
              </div>
            )}
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex items-center justify-between mt-8 pt-6 border-t">
          <button
            type="button"
            onClick={handleBack}
            disabled={step === 1}
            className="px-6 py-3 text-gray-700 font-medium rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Back
          </button>

          {step < 5 ? (
            <button
              type="button"
              onClick={handleNext}
              className="flex items-center space-x-2 bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700"
            >
              <span>Next</span>
              <ChevronRight className="w-5 h-5" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={createExperimentMutation.isPending}
              className="flex items-center space-x-2 bg-green-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-green-700 disabled:opacity-50"
            >
              {createExperimentMutation.isPending ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Creating...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Create & Start Optimization</span>
                </>
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

export default CreateExperiment;