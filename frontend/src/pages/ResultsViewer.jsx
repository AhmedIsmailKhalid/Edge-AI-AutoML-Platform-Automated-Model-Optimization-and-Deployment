/* eslint-disable no-unused-vars */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Calendar, Cpu, Filter, Search, Target, TrendingUp } from 'lucide-react';

import { api } from '../api/client';

export default function ResultsViewer() {
  const navigate = useNavigate();

  const [experiments, setExperiments] = useState([]);
  const [filteredExperiments, setFilteredExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({
    framework: 'all',
    status: 'all',
    device: 'all',
    goal: 'all',
  });

  const fetchAllExperiments = async () => {
    try {
      setLoading(true);
      console.log('🔍 Fetching experiments...');
      const response = await api.getRecentExperiments(100);
      console.log('📦 Received experiments:', response.data);
      console.log('📊 Total experiments:', response.data.length);
      setExperiments(response.data);
    } catch (error) {
      console.error('❌ Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllExperiments();
  }, []);

  useEffect(() => {
    let filtered = [...experiments];

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (exp) =>
          exp.name?.toLowerCase().includes(query) ||
          exp.description?.toLowerCase().includes(query) ||
          exp.model_name?.toLowerCase().includes(query)
      );
    }

    if (filters.framework !== 'all') {
      filtered = filtered.filter((exp) => exp.framework === filters.framework);
    }

    if (filters.status !== 'all') {
      filtered = filtered.filter((exp) => exp.status === filters.status);
    }

    if (filters.device !== 'all') {
      filtered = filtered.filter((exp) => exp.target_device === filters.device);
    }

    if (filters.goal !== 'all') {
      filtered = filtered.filter((exp) => exp.optimization_goal === filters.goal);
    }

    setFilteredExperiments(filtered);
  }, [searchQuery, filters, experiments]);

  const handleFilterChange = (filterType, value) => {
    setFilters((prev) => ({ ...prev, [filterType]: value }));
  };

  const getStatusBadge = (status) => {
    const statusStyles = {
      pending: 'bg-gray-100 text-gray-800',
      running: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800',
    };

    return (
      <span
        className={`px-2 py-1 text-xs font-medium rounded-full ${
          statusStyles[status] || statusStyles.pending
        }`}
      >
        {status ? status.charAt(0).toUpperCase() + status.slice(1) : 'Pending'}
      </span>
    );
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto" />
          <p className="mt-4 text-gray-600">Loading experiments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">All Experiments</h1>
          <p className="mt-2 text-gray-600">
            Browse and search through {experiments.length} experiments
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
              <input
                type="text"
                placeholder="Search by name, description, or model..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Filter className="h-4 w-4" />
              <span className="font-medium">Filters:</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Framework</label>
                <select
                  value={filters.framework}
                  onChange={(e) => handleFilterChange('framework', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Frameworks</option>
                  <option value="pytorch">PyTorch</option>
                  <option value="tensorflow">TensorFlow</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select
                  value={filters.status}
                  onChange={(e) => handleFilterChange('status', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Statuses</option>
                  <option value="completed">Completed</option>
                  <option value="running">Running</option>
                  <option value="failed">Failed</option>
                  <option value="pending">Pending</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Device</label>
                <select
                  value={filters.device}
                  onChange={(e) => handleFilterChange('device', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Devices</option>
                  <option value="raspberry_pi_3b">Raspberry Pi 3B</option>
                  <option value="raspberry_pi_4">Raspberry Pi 4</option>
                  <option value="raspberry_pi_5">Raspberry Pi 5</option>
                  <option value="jetson_nano">Jetson Nano</option>
                  <option value="jetson_xavier_nx">Jetson Xavier NX</option>
                  <option value="coral_dev_board">Coral Dev Board</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <select
                  value={filters.goal}
                  onChange={(e) => handleFilterChange('goal', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Goals</option>
                  <option value="balanced">Balanced</option>
                  <option value="maximize_accuracy">Maximize Accuracy</option>
                  <option value="minimize_size">Minimize Size</option>
                  <option value="minimize_latency">Minimize Latency</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm overflow-hidden">
          {filteredExperiments.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500">No experiments found matching your criteria.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Experiment
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Framework
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Device
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Goal
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>

                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredExperiments.map((experiment) => (
                    <tr
                      key={experiment.id}
                      className="hover:bg-gray-50 cursor-pointer"
                      onClick={() => {
                        if (experiment.status === 'completed') {
                          navigate(`/experiment/${experiment.id}/results`);
                        } else if (experiment.status === 'running') {
                          navigate(`/experiment/${experiment.id}/progress`);
                        }
                      }}
                    >
                      <td className="px-6 py-4">
                        <div className="text-sm font-medium text-gray-900">{experiment.name}</div>
                        {experiment.model_name && (
                          <div className="text-sm text-gray-500">{experiment.model_name}</div>
                        )}
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-900">
                          <Cpu className="h-4 w-4 mr-1" />
                          {experiment.framework}
                        </div>
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-900">
                          <Target className="h-4 w-4 mr-1" />
                          <span className="capitalize">
                            {(experiment.target_device || '').replace(/_/g, ' ')}
                          </span>
                        </div>
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-900">
                          <TrendingUp className="h-4 w-4 mr-1" />
                          <span className="capitalize">
                            {(experiment.optimization_goal || '').replace(/_/g, ' ')}
                          </span>
                        </div>
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap">
                        {getStatusBadge(experiment.status)}
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-500">
                          <Calendar className="h-4 w-4 mr-1" />
                          {formatDate(experiment.created_at)}
                        </div>
                      </td>

                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (experiment.status === 'completed') {
                              navigate(`/experiment/${experiment.id}/results`);
                            } else if (experiment.status === 'running') {
                              navigate(`/experiment/${experiment.id}/progress`);
                            }
                          }}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
