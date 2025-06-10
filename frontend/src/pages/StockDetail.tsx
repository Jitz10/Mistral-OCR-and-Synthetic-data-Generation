import React from "react";
import Layout from "@/components/layout/Layout";
import UploadFileBox from "@/UploadFileBox";

const StockDetail = () => {
  return (
    <Layout>
      <div className="w-full px-4 py-8">
        <div className="space-y-6">
          <UploadFileBox />
        </div>
      </div>
    </Layout>
  );
};

export default StockDetail;