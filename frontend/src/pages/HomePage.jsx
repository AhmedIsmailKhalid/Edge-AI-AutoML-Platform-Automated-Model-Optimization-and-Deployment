/* eslint-disable no-unused-vars */
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Zap, Plus, Search, TrendingUp } from 'lucide-react';
import { api } from '../api/client';
import ExperimentCard from '../components/ExperimentCard';
import LoadingSpinner from '../components/LoadingSpinner';

function HomePage() {
  const navigate = useNavigate();

  // Fetch recent experiments
  const { data: experiments, isLoading } = useQuery({
    queryKey: ['recentExperiments'],
    queryFn: async () => {
      const response = await api.getRecentExperiments(5);
      return response.data;
    },
  });

  return (
    <div className="max-w-7xl mx-auto space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-6 py-12">
        <div className="flex items-center justify-center space-x-3">
          <div className="bg-primary-600 p-3 rounded-xl">
            <Zap className="w-12 h-12 text-white" />
          </div>
        </div>

        <h1 className="text-5xl font-bold text-gray-900">Edge AI AutoML Platform</h1>

        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Intelligent model optimization for edge devices. Automate weeks of manual work into hours
          with AI-powered compression techniques.
        </p>

        {/* Action Buttons */}
        <div className="flex items-center justify-center space-x-4 pt-4">
          <button
            onClick={() => navigate('/create')}
            className="flex items-center space-x-2 bg-primary-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-lg hover:shadow-xl"
          >
            <Plus className="w-5 h-5" />
            <span>Create New Experiment</span>
          </button>

          <button
            onClick={() => navigate('/results')}
            className="flex items-center space-x-2 bg-white text-gray-700 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-colors border-2 border-gray-200"
          >
            <Search className="w-5 h-5" />
            <span>View All Results</span>
          </button>
        </div>
      </div>

      {/* Stats Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-100 p-3 rounded-lg">
              <TrendingUp className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">6</p>
              <p className="text-sm text-gray-600">Optimization Techniques</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="flex items-center space-x-3">
            <div className="bg-green-100 p-3 rounded-lg">
              <Zap className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">2</p>
              <p className="text-sm text-gray-600">Supported Frameworks</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="flex items-center space-x-3">
            <div className="bg-purple-100 p-3 rounded-lg">
              <TrendingUp className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">6</p>
              <p className="text-sm text-gray-600">Target Edge Devices</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Experiments Section */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-3xl font-bold text-gray-900">Recent Experiments</h2>
          <button
            onClick={() => navigate('/results')}
            className="text-primary-600 hover:text-primary-700 font-medium"
          >
            View All →
          </button>
        </div>

        {isLoading ? (
          <div className="py-12">
            <LoadingSpinner text="Loading recent experiments..." />
          </div>
        ) : experiments && experiments.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {experiments.map((experiment) => (
              <ExperimentCard
                key={experiment.id}
                experiment={experiment}
                onClick={() => navigate(`/experiment/${experiment.id}/results`)}
              />
            ))}
          </div>
        ) : (
          <div className="bg-white border-2 border-dashed border-gray-300 rounded-xl p-12 text-center">
            <Zap className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No experiments yet</h3>
            <p className="text-gray-600 mb-6">
              Create your first experiment to get started with model optimization
            </p>
            <button
              onClick={() => navigate('/create')}
              className="bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700 transition-colors"
            >
              Create Experiment
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default HomePage;
