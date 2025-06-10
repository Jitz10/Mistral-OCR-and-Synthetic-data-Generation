import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const LoadingSpinner = () => (
  <div className="loading-spinner">
    <div className="spinner-ring">
      <div></div>
      <div></div>
      <div></div>
      <div></div>
    </div>
    <span className="loading-text">Processing...</span>
  </div>
);

const UploadFileBox = () => {
  const [companyName, setCompanyName] = useState('');
  const [companyTicker, setCompanyTicker] = useState('');
  const [annualReports, setAnnualReports] = useState([]);
  const [fundReports, setFundReports] = useState([]);
  const [investorPresentations, setInvestorPresentations] = useState([]);
  const [concallTranscripts, setConcallTranscripts] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (event, setFiles) => {
    const files = Array.from(event.target.files);
    setFiles(prevFiles => [...prevFiles, ...files]);
  };

  const handleUpload = async () => {
    if (!companyName || !companyTicker) {
      alert('Please fill in company name and ticker');
      return;
    }

    try {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('companyName', companyName);
      formData.append('companyTicker', companyTicker);

      // Append all files with their respective types
      annualReports.forEach((file, index) => {
        formData.append('annualReports', file);
      });
      fundReports.forEach((file, index) => {
        formData.append('fundReports', file);
      });
      investorPresentations.forEach((file, index) => {
        formData.append('investorPresentations', file);
      });
      concallTranscripts.forEach((file, index) => {
        formData.append('concallTranscripts', file);
      });

      const response = await fetch('http://localhost:3000/uploadReport', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log('Upload response:', data);
      
      localStorage.setItem('uploadReportResponse', JSON.stringify(data.data));
      
      // Clear all states
      setCompanyName('');
      setCompanyTicker('');
      setAnnualReports([]);
      setFundReports([]);
      setInvestorPresentations([]);
      setConcallTranscripts([]);

      navigate('/dashboard');
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (fileIndex, fileList, setFileList) => {
    setFileList(fileList.filter((_, index) => index !== fileIndex));
  };

  const FileUploadSection = ({ title, files, setFiles, accept }) => (
    <div className="file-upload-section">
      <h3>{title}</h3>
      <div className="file-list">
        {files.map((file, index) => (
          <div key={index} className="file-item">
            <span>{file.name}</span>
            <button 
              onClick={() => removeFile(index, files, setFiles)}
              className="remove-file-btn"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <div className="file-input-container">
        <input
          type="file"
          multiple
          accept={accept}
          onChange={(e) => handleFileChange(e, setFiles)}
          className="file-input"
          id={`file-input-${title.toLowerCase().replace(/\s+/g, '-')}`}
        />
        <label 
          htmlFor={`file-input-${title.toLowerCase().replace(/\s+/g, '-')}`}
          className="upload-label"
        >
          <div className="upload-content">
            <svg
              className="upload-icon"
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="upload-text">Click to add files</p>
          </div>
        </label>
      </div>
    </div>
  );

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Company Documents</h2>
      
      <div className="main-content">
        <div className="input-section">
          <div className="input-group">
            <label htmlFor="companyName">Company Name</label>
            <input
              type="text"
              id="companyName"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="Enter company name"
              required
            />
          </div>
          
          <div className="input-group">
            <label htmlFor="companyTicker">Company Ticker (NSE/BSE)</label>
            <input
              type="text"
              id="companyTicker"
              value={companyTicker}
              onChange={(e) => setCompanyTicker(e.target.value)}
              placeholder="Enter company ticker"
              required
            />
          </div>
        </div>

        <div className="file-sections">
          <FileUploadSection
            title="Annual Reports"
            files={annualReports}
            setFiles={setAnnualReports}
            accept=".pdf,.doc,.docx"
          />
          
          <FileUploadSection
            title="Fund Reports"
            files={fundReports}
            setFiles={setFundReports}
            accept=".pdf,.doc,.docx"
          />
          
          <FileUploadSection
            title="Investor Presentations"
            files={investorPresentations}
            setFiles={setInvestorPresentations}
            accept=".pdf,.ppt,.pptx"
          />
          
          <FileUploadSection
            title="Concall Transcripts"
            files={concallTranscripts}
            setFiles={setConcallTranscripts}
            accept=".pdf,.doc,.docx,.txt"
          />
        </div>
      </div>

      <button 
        className="upload-button"
        onClick={handleUpload}
        disabled={isUploading || (!companyName || !companyTicker)}
      >
        {isUploading ? <LoadingSpinner /> : 'Upload All Files'}
      </button>

      <style jsx>{`
        .upload-container {
          width: 100%;
          min-height: 100vh;
          margin: 0;
          padding: 2.5rem;
          background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        }

        .upload-title {
          font-size: 2rem;
          font-weight: 700;
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 2rem;
          text-align: center;
          letter-spacing: -0.025em;
        }

        .main-content {
          display: grid;
          grid-template-columns: 1fr 2fr;
          gap: 2rem;
          margin-bottom: 5rem;
        }

        .input-section {
          background: rgba(255, 255, 255, 0.9);
          padding: 1.5rem;
          border-radius: 16px;
          border: 1px solid rgba(99, 102, 241, 0.2);
          height: fit-content;
          position: sticky;
          top: 2rem;
          backdrop-filter: blur(10px);
          box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        }

        .input-group {
          margin-bottom: 1.25rem;
        }

        .input-group:last-child {
          margin-bottom: 0;
        }

        .input-group label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 600;
          color: #4f46e5;
          font-size: 0.95rem;
        }

        .input-group input {
          width: 100%;
          padding: 0.875rem 1rem;
          border: 2px solid rgba(99, 102, 241, 0.2);
          border-radius: 12px;
          font-size: 1rem;
          transition: all 0.3s ease;
          background: #ffffff;
          color: #1e293b;
        }

        .input-group input:focus {
          outline: none;
          border-color: #7c3aed;
          box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.1);
        }

        .input-group input::placeholder {
          color: #94a3b8;
        }

        .file-sections {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1.5rem;
        }

        @media (max-width: 1024px) {
          .main-content {
            grid-template-columns: 1fr;
          }

          .input-section {
            position: static;
          }
        }

        @media (max-width: 768px) {
          .file-sections {
            grid-template-columns: 1fr;
          }
        }

        .file-upload-section {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 16px;
          padding: 1.5rem;
          border: 1px solid rgba(99, 102, 241, 0.2);
          transition: all 0.3s ease;
          height: 100%;
          display: flex;
          flex-direction: column;
          backdrop-filter: blur(10px);
          box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        }

        .file-upload-section:hover {
          border-color: #7c3aed;
          box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
          transform: translateY(-2px);
        }

        .file-upload-section h3 {
          margin: 0 0 1.25rem 0;
          font-size: 1.125rem;
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          font-weight: 600;
        }

        .file-list {
          flex: 1;
          margin-bottom: 1.25rem;
          max-height: 150px;
          overflow-y: auto;
          padding-right: 0.5rem;
        }

        .file-list::-webkit-scrollbar {
          width: 6px;
        }

        .file-list::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 3px;
        }

        .file-list::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 3px;
        }

        .file-list::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }

        .file-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.875rem 1rem;
          background: rgba(99, 102, 241, 0.05);
          border-radius: 12px;
          margin-bottom: 0.75rem;
          border: 1px solid rgba(99, 102, 241, 0.1);
          transition: all 0.2s ease;
        }

        .file-item:hover {
          background: rgba(99, 102, 241, 0.1);
          border-color: rgba(99, 102, 241, 0.2);
        }

        .file-item span {
          flex: 1;
          margin-right: 1rem;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: #1e293b;
          font-size: 0.95rem;
        }

        .remove-file-btn {
          background: none;
          border: none;
          color: #db2777;
          font-size: 1.25rem;
          cursor: pointer;
          padding: 0.25rem 0.5rem;
          border-radius: 6px;
          transition: all 0.2s ease;
        }

        .remove-file-btn:hover {
          background: rgba(219, 39, 119, 0.1);
          color: #be185d;
        }

        .file-input-container {
          position: relative;
          width: 100%;
        }

        .file-input {
          position: absolute;
          width: 100%;
          height: 100%;
          top: 0;
          left: 0;
          opacity: 0;
          cursor: pointer;
        }

        .upload-label {
          display: block;
          cursor: pointer;
          padding: 1.5rem;
          border: 2px dashed rgba(99, 102, 241, 0.3);
          border-radius: 12px;
          transition: all 0.3s ease;
          background: rgba(99, 102, 241, 0.02);
        }

        .upload-label:hover {
          border-color: #7c3aed;
          background: rgba(124, 58, 237, 0.05);
        }

        .upload-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
        }

        .upload-icon {
          color: #7c3aed;
          transition: all 0.3s ease;
        }

        .upload-label:hover .upload-icon {
          transform: translateY(-2px);
          color: #4f46e5;
        }

        .upload-text {
          margin: 0;
          color: #4f46e5;
          font-size: 1rem;
          font-weight: 500;
        }

        .loading-spinner {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.75rem;
        }

        .spinner-ring {
          display: inline-block;
          position: relative;
          width: 24px;
          height: 24px;
        }

        .spinner-ring div {
          box-sizing: border-box;
          display: block;
          position: absolute;
          width: 24px;
          height: 24px;
          border: 3px solid #fff;
          border-radius: 50%;
          animation: spinner-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
          border-color: #fff transparent transparent transparent;
        }

        .spinner-ring div:nth-child(1) {
          animation-delay: -0.45s;
        }

        .spinner-ring div:nth-child(2) {
          animation-delay: -0.3s;
        }

        .spinner-ring div:nth-child(3) {
          animation-delay: -0.15s;
        }

        .loading-text {
          font-size: 0.875rem;
          font-weight: 500;
          color: #ffffff;
          letter-spacing: 0.025em;
        }

        @keyframes spinner-ring {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }

        .upload-button {
          position: fixed;
          bottom: 2rem;
          left: 50%;
          transform: translateX(-50%);
          width: calc(100% - 5rem);
          max-width: 1200px;
          margin: 0;
          padding: 1rem;
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 1.125rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
          z-index: 1000;
          min-height: 3.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .upload-button:hover:not(:disabled) {
          transform: translateX(-50%) translateY(-1px);
          box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
        }

        .upload-button:active:not(:disabled) {
          transform: translateX(-50%) translateY(0);
        }

        .upload-button:disabled {
          background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
          cursor: not-allowed;
          box-shadow: none;
        }

        @media (max-width: 640px) {
          .upload-container {
            margin: 1rem;
            padding: 1.5rem;
          }

          .upload-title {
            font-size: 1.5rem;
          }

          .input-section {
            padding: 1.25rem;
          }

          .file-upload-section {
            padding: 1.25rem;
          }

          .upload-label {
            padding: 1.25rem;
          }

          .upload-button {
            padding: 0.875rem;
            font-size: 1rem;
          }
        }
      `}</style>
    </div>
  );
};

export default UploadFileBox;
