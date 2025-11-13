import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface StitchResponse {
  success: boolean;
  result_image: string;
  message?: string;
}

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const stitchImages = async (images: File[]): Promise<StitchResponse> => {
  const formData = new FormData();
  images.forEach((image, index) => {
    formData.append('images', image);
  });

  const response = await api.post<StitchResponse>('/stitch', formData);
  return response.data;
};
