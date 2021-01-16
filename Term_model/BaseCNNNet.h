#ifndef BASECNNNET_H
#define BASECNNNET_H

#include "BasicType.h"
#include "WeichiMove.h"
#include "Point.h"
#include "WeichiCNNFeatureType.h"

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::vector;
typedef boost::shared_ptr< caffe::MemoryDataLayer<float> > MemoryDataLayerSp;
#elif USE_CAFFE2
#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/tensor.h"
#endif

class CNNNetParam {
public:
	int m_iRandomNum;
	string m_sPrototxt;
	string m_sCaffeModel;
	string m_sThreadBindGpu;
	WeichiCNNNetType m_netType;
	WeichiCNNFeatureType m_featureType;

	CNNNetParam() {}
	CNNNetParam( const string& sParam ) { parse(sParam); }

	bool parse( const string& sParam )
	{
		// param format: net(SL/BV/VN/RL) | random(0~8) | feature(F16/F17/G49/G50) | prototxt | model | thread_bind_gpu
		vector<string> vArgument = splitToVector(sParam,'|');
		if( vArgument.size()!=6 ) { return false; }
		for( int i=0; i<vArgument.size()-1; i++ ) { trim(vArgument[i]); }	// last argument should not trim space

		m_netType = getWeichiCNNNetType(vArgument[0]);
		m_iRandomNum = atoi(vArgument[1].c_str());
		m_featureType = getWeichiCNNFeatureType(vArgument[2]);
		m_sPrototxt = WeichiConfigure::dcnn_dir + vArgument[3];
		m_sCaffeModel = WeichiConfigure::dcnn_dir + vArgument[4];
		m_sThreadBindGpu = vArgument[5];

		return true;
	}
};

class BaseCNNNet
{
protected:
	// input data
	float* m_input;
	float* m_dummy;
	int m_current_batch_size;

	// net info
	Color* m_color;

	// network param
	int m_gpu_id;
	int m_batch_size;
	int m_channel_size;
	int m_input_data_size;
	CNNNetParam m_param;

	// load Net
	bool m_bHasLoadNetWork;

	// random rotation
	Vector<SymmetryType,SYMMETRY_SIZE> m_vSymmetryEntry;
	SymmetryType* m_reverseSymmetric;

#ifdef USE_CAFFE
	boost::shared_ptr< Net<float> > m_caffe_net;
	MemoryDataLayerSp m_data_input_layer;
#elif USE_CAFFE2
	caffe2::TensorCPU m_inputCPU, m_outputSLCPU, m_outputVNCPU;
	caffe2::TensorCUDA m_inputCUDA;
	caffe2::Workspace m_caffe2_workspace;
	caffe2::NetDef m_caffe2_net, m_caffe2_weight;
#endif

public:
	BaseCNNNet( CNNNetParam param )
		: m_input(nullptr), m_dummy(nullptr), m_color(nullptr), m_reverseSymmetric(nullptr)
		, m_batch_size(0), m_channel_size(0), m_input_data_size(0)
	{
		m_param = param;
		m_bHasLoadNetWork = false;
		m_channel_size = m_current_batch_size = 0;
		m_gpu_id = WeichiConfigure::dcnn_default_gpu_device;

		for(int i=0; i<SYMMETRY_SIZE; i++) { m_vSymmetryEntry.push_back(static_cast<SymmetryType>(i)); }
	}

	virtual ~BaseCNNNet()
	{
		if( m_input!=nullptr ) { delete[] m_input; }
		if( m_dummy!=nullptr ) { delete[] m_dummy; }
		if( m_color!=nullptr ) { delete[] m_color; }
		if( m_reverseSymmetric!=nullptr ) { delete[] m_reverseSymmetric; }
	}

	void loadNetWork( bool bDisplayErrorMessage=false )
	{
		if (m_bHasLoadNetWork) { return; }
#ifdef USE_CAFFE
		caffe::Net<float>* net = new caffe::Net<float>(m_param.m_sPrototxt.c_str(), caffe::TEST);
		boost::shared_ptr< Net<float> > tmp(net);
		m_caffe_net = tmp;
		if (m_param.m_sCaffeModel.c_str() != "") { m_caffe_net->CopyTrainedLayersFrom(m_param.m_sCaffeModel.c_str()); }

		// load batch size & channel size
		int height = m_caffe_net->blob_by_name("data")->height();
		int width = m_caffe_net->blob_by_name("data")->	width();
		m_batch_size = m_caffe_net->blob_by_name("data")->num();
		m_channel_size = m_caffe_net->blob_by_name("data")->channels();
		m_input_data_size = m_channel_size * height * width;
		if( CNN_CHANNEL_SIZE[m_param.m_featureType]!=m_channel_size && bDisplayErrorMessage ) {
			CERR() << "[Error] Feature channel size mismatch" << endl;
			exit(0);
		}

		// load data input layer
		m_data_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(m_caffe_net->layer_by_name("data_input_layer"));
		assert(m_data_input_layer);
#elif USE_CAFFE2
		CAFFE_ENFORCE(ReadProtoFromFile(m_param.m_sPrototxt, &m_caffe2_net));
		CAFFE_ENFORCE(ReadProtoFromFile(m_param.m_sCaffeModel, &m_caffe2_weight));

		m_batch_size = WeichiConfigure::dcnn_default_batch_size;
		m_channel_size = CNN_CHANNEL_SIZE[m_param.m_featureType];
		m_input_data_size = m_channel_size * WeichiConfigure::BoardSize * WeichiConfigure::BoardSize;

		// set gpu id & load network
		m_caffe2_net.mutable_device_option()->set_device_type(caffe2::CUDA);
		m_caffe2_net.mutable_device_option()->set_cuda_gpu_id(m_gpu_id);
		m_caffe2_weight.mutable_device_option()->set_device_type(caffe2::CUDA);
		m_caffe2_weight.mutable_device_option()->set_cuda_gpu_id(m_gpu_id);

		CAFFE_ENFORCE(m_caffe2_workspace.RunNetOnce(m_caffe2_weight));
		CAFFE_ENFORCE(m_caffe2_workspace.CreateNet(m_caffe2_net));

		m_inputCPU.Resize(vector<caffe2::TIndex>({ m_batch_size,m_channel_size,WeichiConfigure::BoardSize,WeichiConfigure::BoardSize }));
		m_outputSLCPU.Resize(vector<caffe2::TIndex>({ m_batch_size,m_channel_size,WeichiConfigure::BoardSize,WeichiConfigure::BoardSize }));
		m_outputVNCPU.Resize(vector<caffe2::TIndex>({ m_batch_size,m_channel_size,WeichiConfigure::BoardSize,WeichiConfigure::BoardSize }));
		m_inputCUDA.Resize(vector<caffe2::TIndex>({ m_batch_size,m_channel_size,WeichiConfigure::BoardSize,WeichiConfigure::BoardSize }));
		m_caffe2_workspace.GetBlob("data")->template GetMutable<caffe2::TensorCUDA>()->ResizeLike(m_inputCUDA);
#endif
		// malloc cpu array data
		m_color = new Color[m_batch_size];
		m_reverseSymmetric = new SymmetryType[m_batch_size];
		m_input = new float[m_batch_size*m_input_data_size];
		m_dummy = new float[m_batch_size*m_input_data_size];

		m_bHasLoadNetWork = true;
	}

	inline void push_back( const float* data, Color turnColor, SymmetryType type=SYM_NORMAL )
	{
		if( getRandomNum()==0 ) {
			setBatchData(data,turnColor,type);
		} else {
			for(int i=0; i<getRandomNum(); i++ ) {
				int index = Random::nextInt(SYMMETRY_SIZE-i);
				setBatchData(data,turnColor,m_vSymmetryEntry[index]);
				m_vSymmetryEntry.swap(index,SYMMETRY_SIZE-1-i);
			}
		}
	}

	inline void setBatchData( const float* data, const Color turnColor, SymmetryType type=SYM_NORMAL )
	{
		m_reverseSymmetric[m_current_batch_size] = ReverseSymmetricType[type];
		
		int start_index = m_current_batch_size*m_input_data_size;
		memcpy((m_input+start_index),data,sizeof(float)*m_input_data_size);
		m_color[m_current_batch_size] = turnColor;

		rotateFeature((m_input+start_index),type);
		m_current_batch_size++;
	}

	bool reloadNetWork( string sCaffeModel )
	{
		fstream fin(sCaffeModel.c_str(),ios::in);

		if (!fin) {
			CERR() << "[Error] \"" << sCaffeModel << "\" is not exist." << endl;
			return false;
		}

		m_param.m_sCaffeModel = sCaffeModel;
#ifdef USE_CAFFE
		if( !m_bHasLoadNetWork ) { loadNetWork(); }
		else { m_caffe_net->CopyTrainedLayersFrom(m_param.m_sCaffeModel.c_str()); }
#elif USE_CAFFE2
		if( !m_bHasLoadNetWork ) { loadNetWork(); }
		else {
			caffe2::NetDef initDef; 
			//reload init net
			CAFFE_ENFORCE(ReadProtoFromFile(sCaffeModel, &initDef));
			initDef.mutable_device_option()->set_device_type(caffe2::CUDA);
			initDef.mutable_device_option()->set_cuda_gpu_id(m_gpu_id);
			m_caffe2_workspace.RunNetOnce(initDef);
			m_caffe2_workspace.GetBlob("data")->template GetMutable<caffe2::TensorCUDA>()->ResizeLike(m_inputCUDA);
		}
#endif
		return true;
	}

#ifdef USE_CAFFE
	void reloadNetWorkByParam(caffe::NetParameter &net_param) { m_caffe_net->CopyTrainedLayersFrom(net_param); }
	void netParamToProto(caffe::NetParameter& net_param) { m_caffe_net->ToProto(&net_param); }
#endif

	void saveNetWork( string sSnapshot )
	{
#ifdef USE_CAFFE
		caffe::NetParameter net_param;
		m_caffe_net->ToProto(&net_param);
		caffe::WriteProtoToBinaryFile(net_param,sSnapshot.c_str());
#elif USE_CAFFE2
		caffe2::NetDef init_net_;
		std::vector<string> op_map;
		
		caffe2::OperatorDef *data_op = init_net_.add_op();
		data_op->add_arg()->CopyFrom(caffe2::MakeArgument("shape", vector<int>(1)));
		data_op->set_type("GaussianFill");
		data_op->add_output("data");


		for (auto inputName : m_caffe2_workspace.GetNet("train")->external_input()) {
			caffe2::TensorCPU tensor;
			std::string blob_name = inputName.substr(6);

			if (find(op_map.begin(), op_map.end(), blob_name) == op_map.end() && inputName.find("allreduce") == string::npos
				&& blob_name.find("label") == string::npos && blob_name.find("data") == string::npos) {
				op_map.push_back(blob_name);
				if (blob_name.find("iter") != string::npos) {
					tensor.CopyFrom(m_caffe2_workspace.GetBlob(inputName)->Get<caffe2::TensorCPU>());
				} else {
					tensor.CopyFrom(m_caffe2_workspace.GetBlob(inputName)->Get<caffe2::TensorCUDA>());
				}
				caffe2::OperatorDef *op = init_net_.add_op();
				// shape
				op->add_arg()->CopyFrom(caffe2::MakeArgument("shape", tensor.dims()));
				if (blob_name.find("iter") != string::npos) {
					op->set_type("ConstantFill");
					const long *weights = tensor.data<long>();
					std::vector<long> weightsVector(weights, weights + tensor.size());
					op->add_arg()->CopyFrom(caffe2::MakeArgument("dtype", 10));
					op->add_arg()->CopyFrom(caffe2::MakeArgument("values", weightsVector));
				} else {
					op->set_type("GivenTensorFill");
					const float *weights = tensor.data<float>();
					std::vector<float> weightsVector(weights, weights + tensor.size());
					op->add_arg()->CopyFrom(caffe2::MakeArgument("values", weightsVector));
				}
				// output
				op->add_output(blob_name);
			}
		}
		caffe2::WriteProtoToBinaryFile(init_net_, sSnapshot.c_str());
#endif
	}

	inline void resetCurrentBatchSize() { m_current_batch_size = 0; }
	inline int getCurrentBatchSize() { return m_current_batch_size; }
	inline bool isEmpty() const { return (getSpaceLeft()==m_batch_size); }
	inline bool isFull() const { return (getSpaceLeft()<=0); }
	inline int getSpaceLeft() const {
		if( m_param.m_iRandomNum==0 ) { return (m_batch_size-m_current_batch_size); }
		else { return (m_batch_size/m_param.m_iRandomNum-m_current_batch_size); }
	}
	inline void setRandomNum( int randomNum ) { m_param.m_iRandomNum = randomNum; }
	inline int getRandomNum() { return m_param.m_iRandomNum; }
	inline CNNNetParam getNetParam() { return m_param; }
	inline void setGpuID( uint gpuID ) {
		m_gpu_id = gpuID;
#ifdef USE_CAFFE
		caffe::Caffe::SetDevice(m_gpu_id);
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
	}
	inline int getGpuID() const { return m_gpu_id; }
	inline WeichiCNNNetType getCNNNetType() { return m_param.m_netType; }
	inline WeichiCNNFeatureType getCNNFeatureType() { return m_param.m_featureType; }
	inline int getBatchSize() const { return m_batch_size; }
	inline int getChannelSize() const { return m_channel_size; }
	inline int getInputDataSize() const { return m_input_data_size; }
	inline Color getColor( int batch ) { return m_color[batch]; }
	inline float* getFeatures( int batch ) { return (m_input+batch*m_input_data_size); }
	inline SymmetryType getSymmetryType( int batch ) { return m_reverseSymmetric[batch]; }

	inline void rotateFeature( float features[], SymmetryType type=SYM_NORMAL )
	{
		if( type==SYM_NORMAL ) { return; }

		for( int channel=0; channel<m_channel_size; channel++ ) {
			uint start = WeichiConfigure::TotalGrids*channel;
			rotateOneFeature(features+start,type);
		}
	}

	inline void rotateOneFeature( float features[], SymmetryType type=SYM_NORMAL )
	{
		if( type==SYM_NORMAL ) { return; }

		float rotateFeature[MAX_NUM_GRIDS];
		for( uint pos=0; pos<WeichiConfigure::TotalGrids; pos++ ) {
			uint rotatePos = rotatePosition(pos,type);
			rotateFeature[rotatePos] = features[pos];
		}
		for( uint i=0; i<WeichiConfigure::TotalGrids; i++ ) { features[i] = rotateFeature[i]; }
	}

	inline uint rotatePosition( uint position, SymmetryType type=SYM_NORMAL )
	{
		if( type==SYM_NORMAL ) { return position; }

		const int center = WeichiConfigure::BoardSize/2;
		int x = position % WeichiConfigure::BoardSize;
		int y = position / WeichiConfigure::BoardSize;
		Point point(x-center,y-center);
		Point rotatePoint = point.getSymmetryOf(type);
		return ((rotatePoint.getY()+center)*WeichiConfigure::BoardSize + (rotatePoint.getX()+center));
	}
};

#endif
