// import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { 
  Download, 
  ArrowLeft, 
  CheckCircle, 
  TrendingDown, 
  Zap,
  Award,
  AlertCircle
} from 'lucide-react';
import { api } from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';

function ExperimentResults() {
  const { id } = useParams();
  const navigate = useNavigate();

  // Fetch experiment details
  const { data: experiment, isLoading: experimentLoading } = useQuery({
    queryKey: ['experiment', id],
    queryFn: async () => {
      const response = await api.getExperiment(id);
      return response.data;
    },
  });

  // Fetch results
  const { data: resultsData, isLoading: resultsLoading } = useQuery({
    queryKey: ['results', id],
    queryFn: async () => {
      const response = await api.getResults(id);
      // Ensure we always have an array
      return Array.isArray(response.data) ? response.data : [];
    },
    enabled: !!experiment,
  });

  const results = resultsData || [];

  // Fetch recommendations
  const { data: recommendationsData, isLoading: recommendationsLoading } = useQuery({
    queryKey: ['recommendations', id],
    queryFn: async () => {
      const response = await api.getRecommendations(id);
      // Ensure we always have an array
      return Array.isArray(response.data) ? response.data : [];
    },
    enabled: !!experiment,
  });

  const recommendations = recommendationsData || [];

  const handleDownload = async (resultId, techniqueName) => {
    try {
      const response = await api.downloadModel(id, resultId);
      
      // Create blob and download
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${experiment.name}_${techniqueName}_optimized.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download model. Please try again.');
    }
  };

  if (experimentLoading || resultsLoading || recommendationsLoading) {
    return (
      <div className="max-w-6xl mx-auto py-12">
        <LoadingSpinner text="Loading results..." />
      </div>
    );
  }

  if (!experiment || !results) {
    return (
      <div className="max-w-6xl mx-auto py-12">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <p className="text-red-800">Experiment or results not found.</p>
        </div>
      </div>
    );
  }

  const topRecommendation = recommendations.length > 0 ? recommendations[0] : null;

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {experiment.name}
          </h1>
          <p className="text-gray-600">
            Optimization completed on {new Date(experiment.completed_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center space-x-2 bg-green-50 border border-green-200 px-4 py-2 rounded-lg">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <span className="text-green-800 font-medium">Completed</span>
        </div>
      </div>

      {/* Experiment Info */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-600">Framework</p>
            <p className="font-semibold text-gray-900 capitalize">{experiment.framework}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Dataset</p>
            <p className="font-semibold text-gray-900">{experiment.dataset_name}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Target Device</p>
            <p className="font-semibold text-gray-900 capitalize">
              {experiment.target_device?.replace(/_/g, ' ')}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Optimization Goal</p>
            <p className="font-semibold text-gray-900 capitalize">
              {experiment.optimization_goal?.replace(/_/g, ' ')}
            </p>
          </div>
        </div>
      </div>

      {/* Top Recommendation */}
      {topRecommendation && (
        <div className="bg-gradient-to-r from-primary-50 to-primary-100 rounded-xl border-2 border-primary-200 p-6 mb-6">
          <div className="flex items-start space-x-3">
            <Award className="w-6 h-6 text-primary-600 mt-1" />
            <div className="flex-1">
              <h2 className="text-xl font-bold text-gray-900 mb-2">
                Recommended Configuration
              </h2>
              <p className="text-gray-700 mb-4">{topRecommendation.reasoning}</p>
              
              <div className="grid grid-cols-3 gap-4 bg-white rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600">Technique</p>
                  <p className="font-semibold text-gray-900 capitalize">
                    {topRecommendation.technique_name?.replace(/_/g, ' ')}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Accuracy</p>
                  <p className="font-semibold text-gray-900">
                    {(topRecommendation.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Confidence</p>
                  <p className="font-semibold text-gray-900">
                    {(topRecommendation.confidence_score * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
              
              <button
                onClick={() => handleDownload(topRecommendation.result_id, topRecommendation.technique_name)}
                className="mt-4 inline-flex items-center space-x-2 bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700"
              >
                <Download className="w-5 h-5" />
                <span>Download Recommended Model</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">All Optimization Results</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Technique
                </th>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Accuracy
                </th>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Size Reduction
                </th>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Latency (ms)
                </th>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Status
                </th>
                <th className="text-left px-4 py-3 text-sm font-semibold text-gray-700">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {results.map((result, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">
                    <span className="font-medium capitalize">
                      {result.technique_name?.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-900">
                        {(result.optimized_accuracy * 100).toFixed(1)}%
                      </span>
                      {result.accuracy_drop_percent !== null && (
                        <span className={`text-xs ${
                          result.accuracy_drop_percent > 5 ? 'text-red-600' : 'text-green-600'
                        }`}>
                          ({result.accuracy_drop_percent > 0 ? '-' : '+'}{Math.abs(result.accuracy_drop_percent).toFixed(1)}%)
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {result.size_reduction_percent !== null ? (
                      <div className="flex items-center space-x-2">
                        <TrendingDown className="w-4 h-4 text-green-600" />
                        <span className="text-gray-900">
                          {result.size_reduction_percent.toFixed(1)}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-gray-400">N/A</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {result.inference_latency_ms !== null ? (
                      <div className="flex items-center space-x-2">
                        <Zap className="w-4 h-4 text-blue-600" />
                        <span className="text-gray-900">
                          {result.inference_latency_ms.toFixed(1)}
                        </span>
                      </div>
                    ) : (
                      <span className="text-gray-400">N/A</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {result.status === 'completed' ? (
                      <span className="inline-flex items-center space-x-1 bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                        <CheckCircle className="w-3 h-3" />
                        <span>Success</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                        <AlertCircle className="w-3 h-3" />
                        <span>Failed</span>
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {result.status === 'completed' && (
                      <button
                        onClick={() => handleDownload(result.id, result.technique_name)}
                        className="inline-flex items-center space-x-1 text-primary-600 hover:text-primary-700 font-medium"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download</span>
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Additional Recommendations */}
      {recommendations && recommendations.length > 1 && (
        <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Alternative Recommendations</h2>
          <div className="space-y-4">
            {recommendations.slice(1).map((rec, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 capitalize mb-2">
                      {rec.technique_name?.replace(/_/g, ' ')}
                    </h3>
                    <p className="text-sm text-gray-600 mb-3">{rec.reasoning}</p>
                    <div className="flex items-center space-x-6 text-sm">
                      <div>
                        <span className="text-gray-600">Accuracy: </span>
                        <span className="font-medium text-gray-900">
                          {(rec.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Confidence: </span>
                        <span className="font-medium text-gray-900">
                          {(rec.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDownload(rec.result_id, rec.technique_name)}
                    className="ml-4 text-primary-600 hover:text-primary-700"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ExperimentResults;