import { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import { stitchImages } from './services/api';

function App() {
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [stitchedImage, setStitchedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStitch = async () => {
    if (selectedImages.length < 2) {
      setError('최소 2개 이상의 이미지를 선택해주세요.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setStitchedImage(null);

    try {
      const response = await stitchImages(selectedImages);
      if (response.success) {
        setStitchedImage(response.result_image);
      } else {
        setError(response.message || '이미지 스티칭에 실패했습니다.');
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || '서버 오류가 발생했습니다. 백엔드 서버가 실행 중인지 확인해주세요.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImages([]);
    setStitchedImage(null);
    setError(null);
  };

  const handleDownload = () => {
    if (!stitchedImage) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${stitchedImage}`;
    link.download = `stitched-panorama-${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-800 mb-4">
            AI 이미지 스티칭
          </h1>
          <p className="text-xl text-gray-600">
            딥러닝 기반 자동 파노라마 생성
          </p>
          <p className="text-sm text-gray-500 mt-2">
            SuperGlue Transformer 모델 사용
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <ImageUploader
            onImagesSelected={setSelectedImages}
            selectedImages={selectedImages}
          />

          {error && (
            <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          <div className="mt-8 flex justify-center gap-4">
            <button
              onClick={handleStitch}
              disabled={selectedImages.length < 2 || isLoading}
              className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-lg"
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  처리 중...
                </span>
              ) : (
                '이미지 합성하기'
              )}
            </button>

            {selectedImages.length > 0 && (
              <button
                onClick={handleReset}
                disabled={isLoading}
                className="px-8 py-3 bg-gray-500 text-white font-semibold rounded-lg hover:bg-gray-600 disabled:bg-gray-300 transition-colors shadow-lg"
              >
                초기화
              </button>
            )}
          </div>
        </div>

        {stitchedImage && (
          <div className="bg-white rounded-2xl shadow-2xl p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">합성 결과</h2>
            <div className="relative">
              <img
                src={`data:image/png;base64,${stitchedImage}`}
                alt="Stitched panorama"
                className="w-full rounded-lg shadow-lg"
              />
            </div>
            <div className="mt-6 flex justify-center">
              <button
                onClick={handleDownload}
                className="px-8 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors shadow-lg flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
                다운로드
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
