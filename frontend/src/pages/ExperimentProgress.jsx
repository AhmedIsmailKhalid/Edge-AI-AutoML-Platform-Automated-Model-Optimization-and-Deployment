import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Loader2, CheckCircle, XCircle, Zap } from 'lucide-react';
import { api } from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';

// Technique display names
const techniqueDisplayNames = {
  ptq_int8: 'Post-Training Quantization (INT8)',
  ptq_int4: 'Post-Training Quantization (INT4)',
  pruning_magnitude_unstructured: 'Magnitude Pruning (Unstructured)',
  pruning_magnitude_structured: 'Magnitude Pruning (Structured)',
  quantization_aware_training: 'Quantization-Aware Training',
  hybrid_prune_quantize: 'Hybrid (Pruning + Quantization)',
  distillation: 'Knowledge Distillation',
};

function ExperimentProgress() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [progress, setProgress] = useState(0);
  const [currentTechnique, setCurrentTechnique] = useState(null);
  const [completedTechniques, setCompletedTechniques] = useState([]);
  const [experimentStatus, setExperimentStatus] = useState('running');

  // Get experiment details
  const { data: experiment, isLoading } = useQuery({
    queryKey: ['experiment', id],
    queryFn: async () => {
      const response = await api.getExperiment(id);
      return response.data;
    },
  });

  // Connect to SSE stream
  useEffect(() => {
    console.log('🔵 Connecting to SSE stream for experiment:', id);
    
    const eventSource = new EventSource(
      `http://localhost:8000/api/optimize/${id}/progress-stream`
    );

    eventSource.onopen = () => {
      console.log('✅ SSE connection established');
    };

    eventSource.onmessage = (event) => {
      console.log('📨 SSE message received:', event.data);
      
      try {
        const data = JSON.parse(event.data);
        console.log('📦 Parsed data:', data);
        
        if (data.type === 'technique_update') {
          const displayName = techniqueDisplayNames[data.technique] || data.technique;
          
          console.log('🔧 Technique update:', data.status, displayName);
          
          if (data.status === 'running') {
            console.log('Setting current technique:', displayName);
            setCurrentTechnique({
              name: displayName,
              techniqueName: data.technique,
              status: 'running',
            });
            console.log('✅ State set, should trigger re-render');
          } else if (data.status === 'completed') {
            console.log('Technique completed:', displayName);
            setCompletedTechniques((prev) => [
              ...prev,
              {
                name: displayName,
                techniqueName: data.technique,
                accuracy: data.accuracy,
                sizeReduction: data.size_reduction,
              },
            ]);
            // setCurrentTechnique(null);
          } else if (data.status === 'failed') {
            console.log('Technique failed:', displayName);
            setCompletedTechniques((prev) => [
              ...prev,
              {
                name: displayName,
                techniqueName: data.technique,
                status: 'failed',
              },
            ]);
            setCurrentTechnique(null);
          }
        } else if (data.type === 'progress') {
          console.log('📊 Progress update:', data.progress);
          setProgress(data.progress);
        } else if (data.type === 'complete') {
          console.log('🎉 Optimization complete!');
          setExperimentStatus(data.status);
          
          setTimeout(() => {
            navigate(`/experiment/${id}/results`);
          }, 2000);
        }
      } catch (error) {
        console.error('❌ Error parsing SSE data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('❌ SSE connection error:', error);
      console.log('EventSource readyState:', eventSource.readyState);
      eventSource.close();
    };

    return () => {
      console.log('🔴 Closing SSE connection');
      eventSource.close();
    };
  }, [id, navigate]);

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto py-12">
        <LoadingSpinner text="Loading experiment..." />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          {experiment?.name}
        </h1>
        <p className="text-gray-600">
          Real-time optimization progress
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-600">Framework</p>
            <p className="font-semibold text-gray-900 capitalize">
              {experiment?.framework}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Dataset</p>
            <p className="font-semibold text-gray-900">
              {experiment?.dataset_name}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Target Device</p>
            <p className="font-semibold text-gray-900 capitalize">
              {experiment?.target_device?.replace(/_/g, ' ')}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Goal</p>
            <p className="font-semibold text-gray-900 capitalize">
              {experiment?.optimization_goal?.replace(/_/g, ' ')}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Optimization Progress
        </h2>

        <div className="mb-8">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>Overall Progress</span>
            <span className="font-semibold">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-primary-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {currentTechnique && (
          <div className="mb-8 p-6 bg-blue-50 border-2 border-blue-200 rounded-xl">
            <div className="flex items-center space-x-3">
              <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
              <div>
                <p className="text-sm text-blue-600 font-medium">
                  Currently Running
                </p>
                <p className="text-lg font-semibold text-gray-900">
                  {currentTechnique.name}
                </p>
              </div>
            </div>
          </div>
        )}

        {completedTechniques.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Completed Techniques
          </h3>
          {completedTechniques.map((technique, index) => (
            <div
              key={index}
              className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg border border-gray-200"
            >
              {technique.status === 'failed' ? (
                <XCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              ) : (
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
              )}
              <div className="flex-1">
                <p className="font-semibold text-gray-900">
                  {technique.name}
                </p>
                {technique.accuracy !== undefined && technique.accuracy !== null && (
                  <p className="text-sm text-gray-600 mt-1">
                    {(technique.accuracy * 100).toFixed(1)}% accuracy
                    {technique.sizeReduction !== null && technique.sizeReduction !== undefined && (
                      <>, {technique.sizeReduction.toFixed(1)}% size reduction</>
                    )}
                  </p>
                )}
                {technique.status === 'failed' && (
                  <p className="text-sm text-red-600 mt-1">Failed</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

        {experimentStatus === 'completed' && (
          <div className="mt-8 p-6 bg-green-50 border-2 border-green-200 rounded-xl">
            <div className="flex items-center space-x-3">
              <CheckCircle className="w-8 h-8 text-green-600" />
              <div>
                <p className="text-lg font-semibold text-gray-900">
                  Optimization Complete!
                </p>
                <p className="text-sm text-gray-600 mt-1">
                  Redirecting to results page...
                </p>
              </div>
            </div>
          </div>
        )}

        {experimentStatus === 'failed' && (
          <div className="mt-8 p-6 bg-red-50 border-2 border-red-200 rounded-xl">
            <div className="flex items-center space-x-3">
              <XCircle className="w-8 h-8 text-red-600" />
              <div>
                <p className="text-lg font-semibold text-gray-900">
                  Optimization Failed
                </p>
                <p className="text-sm text-gray-600 mt-1">
                  Please check the logs or try again
                </p>
              </div>
            </div>
          </div>
        )}

        {completedTechniques.length === 0 && !currentTechnique && experimentStatus === 'running' && (
          <div className="text-center py-12">
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Initializing optimization...</p>
          </div>
        )}
      </div>

      {experimentStatus === 'completed' && (
        <div className="mt-6 text-center">
          <button
            onClick={() => navigate(`/experiment/${id}/results`)}
            className="inline-flex items-center space-x-2 bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700"
          >
            <Zap className="w-5 h-5" />
            <span>View Results Now</span>
          </button>
        </div>
      )}
    </div>
  );
}

export default ExperimentProgress;