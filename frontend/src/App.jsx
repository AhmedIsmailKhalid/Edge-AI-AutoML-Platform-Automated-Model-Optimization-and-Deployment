import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Pages
import HomePage from './pages/HomePage';
import CreateExperiment from './pages/CreateExperiment';
import ExperimentProgress from './pages/ExperimentProgress';
import ExperimentResults from './pages/ExperimentResults';
import ResultsViewer from './pages/ResultsViewer';

// Components
import Header from './components/Header';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Header />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/create" element={<CreateExperiment />} />
              <Route path="/experiment/:id/progress" element={<ExperimentProgress />} />
              <Route path="/experiment/:id/results" element={<ExperimentResults />} />
              <Route path="/results" element={<ResultsViewer />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;