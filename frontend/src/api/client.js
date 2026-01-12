import axios from 'axios';

// Base URL for API
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with longer timeout
const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for Render cold starts
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🔵 ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    console.log(
      `✅ ${response.config.method.toUpperCase()} ${response.config.url} - ${response.status}`
    );
    return response;
  },
  (error) => {
    console.error(
      `❌ ${error.config?.method?.toUpperCase()} ${error.config?.url} - ${error.response?.status || 'Network Error'}`
    );
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Experiments
  createExperiment: (data) => apiClient.post('/api/experiments/create', data),
  getExperiment: (id) => apiClient.get(`/api/experiments/${id}`),
  getRecentExperiments: (limit = 20) => apiClient.get(`/api/experiments/recent?limit=${limit}`),
  searchExperiments: (params) => apiClient.get('/api/experiments/search', { params }),

  // Upload
  uploadModel: (experimentId, formData) =>
    apiClient.post(`/api/upload/${experimentId}/model`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes for file uploads
    }),
  uploadDataset: (experimentId, formData) =>
    apiClient.post(`/api/upload/${experimentId}/dataset`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes for file uploads
    }),

  // Optimization
  startOptimization: (experimentId) => apiClient.post(`/api/optimize/${experimentId}/start`),

  // Results
  getResults: (experimentId) => apiClient.get(`/api/results/${experimentId}/results`),
  getRecommendations: (experimentId) =>
    apiClient.get(`/api/results/${experimentId}/recommendations`),
  downloadModel: (experimentId, techniqueName) =>
    apiClient.get(`/api/results/${experimentId}/download/${techniqueName}`, {
      responseType: 'blob',
      timeout: 300000, // 5 minutes for downloads
    }),
};

export default apiClient;
