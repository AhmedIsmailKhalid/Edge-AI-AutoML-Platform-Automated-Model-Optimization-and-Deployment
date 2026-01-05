import { Clock, Cpu, CheckCircle, XCircle, Loader2 } from 'lucide-react';

function ExperimentCard({ experiment, onClick }) {
  const getStatusConfig = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bg: 'bg-green-100',
          text: 'Completed',
        };
      case 'running':
        return {
          icon: Loader2,
          color: 'text-blue-600',
          bg: 'bg-blue-100',
          text: 'Running',
          animate: true,
        };
      case 'failed':
        return {
          icon: XCircle,
          color: 'text-red-600',
          bg: 'bg-red-100',
          text: 'Failed',
        };
      default:
        return {
          icon: Clock,
          color: 'text-gray-600',
          bg: 'bg-gray-100',
          text: 'Pending',
        };
    }
  };

  const statusConfig = getStatusConfig(experiment.status);
  const StatusIcon = statusConfig.icon;

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <div
      onClick={onClick}
      className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all cursor-pointer group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors line-clamp-1">
            {experiment.name}
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {formatDate(experiment.created_at)}
          </p>
        </div>

        {/* Status Badge */}
        <div className={`flex items-center space-x-1 px-3 py-1 rounded-full ${statusConfig.bg}`}>
          <StatusIcon
            className={`w-4 h-4 ${statusConfig.color} ${statusConfig.animate ? 'animate-spin' : ''}`}
          />
          <span className={`text-xs font-medium ${statusConfig.color}`}>
            {statusConfig.text}
          </span>
        </div>
      </div>

      {/* Details */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <Cpu className="w-4 h-4" />
          <span className="capitalize">{experiment.framework}</span>
        </div>

        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <span className="font-medium">Dataset:</span>
          <span>{experiment.dataset_name}</span>
        </div>

        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <span className="font-medium">Device:</span>
          <span className="capitalize">
            {experiment.target_device?.replace(/_/g, ' ')}
          </span>
        </div>
      </div>

      {/* Progress */}
      {experiment.status === 'running' && experiment.progress_percent !== undefined && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
            <span>Progress</span>
            <span>{experiment.progress_percent}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${experiment.progress_percent}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default ExperimentCard;