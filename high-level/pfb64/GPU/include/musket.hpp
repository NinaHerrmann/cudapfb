#pragma once
#include <string>
#include "pfb64.hpp"

namespace mkt {
enum Distribution {DIST, COPY};
template<typename T>
class DArray {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, mkt::Distribution d = DIST, Distribution device_dist = DIST);
  ~ DArray();
  
  template<std::size_t N> 
  void operator=(const std::array<T, N>& a);
  
   void update_self();
   void update_devices();
   void map_pointer();
   
// Getter and Setter

  T get_global(int index);
  void set_global(int index, const T& value);

  T get_local(int index);
  void set_local(int index, const T& value);

  T& operator[](int local_index);
  const T& operator[](int local_index) const;

  int get_size() const;
  int get_size_local() const;
  int get_size_gpu() const;

  int get_offset() const;
		
  Distribution get_distribution() const;
  Distribution get_device_distribution() const;
		
  T* get_data();
  const T* get_data() const;
  
  T* get_device_pointer(int gpu) const;
		
 private:

  int get_gpu_by_local_index(int local_index) const;
  int get_gpu_by_global_index(int global_index) const;

  //
  // Attributes
  //

  // position of processor in data parallel group of processors; zero-base
  int _pid;

  int _size;
  int _size_local;
  int _size_gpu;

  // number of (local) partitions in array
  int _partitions;

  // position of processor in data parallel group of processors
  int _partition_pos;

  // first index in local partition
  int _offset;

  // checks whether data is copy distributed among all processes
  Distribution _dist;
  Distribution _device_dist;

  std::vector<T> _data;
  std::array<T*, 1> _host_data;
  std::array<T*, 1> _gpu_data;
};
template<typename T, typename R, typename Functor>
void map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename Functor>
void map_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void map_index_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void map_local_index_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void fold(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);

template<typename T, typename Functor>
void fold_copy(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor& f_map, const I identity, const FoldFunctor f_fold);

template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor f_map, const I identity, const FoldFunctor f_fold);
template<typename T>
class DeviceArray {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DeviceArray(const DArray<T>& da);
  DeviceArray(const DeviceArray<T>& da);
  ~DeviceArray();
  
  void init(int device_id);
  
  
// Getter and Setter

  const T& get_data_device(int device_index) const;

  const T& get_data_local(int local_index) const;

 private:

  //
  // Attributes
  //

  int _size;
  int _size_local;
  int _size_device;

  int _offset;
  
  int _device_offset;

  Distribution _dist;
  Distribution _device_dist;

  T* _device_data;

  std::array<T*, 1> _gpu_data;

  };



template<typename T>
void print(std::ostringstream& stream, const T& a);


template<typename T>
void gather(mkt::DArray<T>& in, mkt::DArray<T>& out);
	
template<typename T>
void scatter(mkt::DArray<T>& in, mkt::DArray<T>& out);
	





} // namespace mkt

template<typename T>
mkt::DArray<T>::DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, Distribution d, Distribution device_dist)
    : _pid(pid),
      _size(size),
      _size_local(size_local),
      _size_gpu(0), 
      _partitions(partitions),
      _partition_pos(partition_pos),
      _offset(offset),
      _dist(d),
      _device_dist(device_dist),
      _data(size_local, init_value) {
    
    if(device_dist == mkt::Distribution::DIST){
    	_size_gpu = size_local / 1; // assume even distribution for now
    }else if(device_dist == mkt::Distribution::COPY){
    	_size_gpu = size_local;
    }
    
    #pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		// allocate memory
		T* devptr = static_cast<T*>(acc_malloc(_size_gpu * sizeof(T)));
		
		// store pointer to device memory and host memory
		_gpu_data[gpu] = devptr;
		if(device_dist == mkt::Distribution::DIST){
	    	_host_data[gpu] = _data.data() + gpu * _size_gpu;
	    }else if(device_dist == mkt::Distribution::COPY){
	    	_host_data[gpu] = _data.data(); // all gpus have complete data, thus point to the beginning of host vector
	    }
		
		
		// init values
		#pragma acc parallel loop deviceptr(devptr) async(0)
		for(unsigned int i = 0; i < _size_gpu; ++i){
		  devptr[i] = init_value;
		}
	}
	this->map_pointer();
}

template<typename T>
mkt::DArray<T>::~DArray(){
	// free device memory
	#pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_free(_gpu_data[gpu]);
	}
}
		
template<typename T>
template<std::size_t N>
void mkt::DArray<T>::operator=(const std::array<T, N>& a) {
  acc_wait(0);
  #pragma omp parallel for
  for(unsigned int element = 0; element < _size_local; ++element){
	_data[element] = a[element];
  }
  update_devices();
}	
		
template<typename T>
void mkt::DArray<T>::update_self() {
  	#pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_update_self_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
		//acc_memcpy_from_device_async(acc_hostptr(_gpu_data[gpu]), _gpu_data[gpu], _size_gpu * sizeof(T), 0);
		acc_wait(0);
	}
}

template<typename T>
void mkt::DArray<T>::update_devices() {
  	#pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_update_device_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
		acc_wait(0);
	}
}

template<typename T>
void mkt::DArray<T>::map_pointer() {
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_map_data( _host_data[gpu], _gpu_data[gpu], _size_gpu * sizeof(T));
		//acc_map_data( _data.data() + gpu * _size_gpu , _gpu_data[gpu], _size_gpu * sizeof(T));
	}
}
		
template<typename T>
T mkt::DArray<T>::get_local(int index) {
	int gpu = get_gpu_by_local_index(index);
	acc_set_device_num(gpu, acc_device_not_host);
	T* host_pointer = _data.data() + index;
	T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
	acc_memcpy_from_device_async(host_pointer, gpu_pointer, sizeof(T), 0);
	acc_wait(0);
    return _data[index];
}

template<typename T>
void mkt::DArray<T>::set_local(int index, const T& v) {
	_data[index] = v;
	T* host_pointer = _data.data() + index;
	if(_device_dist == mkt::Distribution::COPY){
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			T* gpu_pointer = _gpu_data[gpu] + index;
			acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
		}
	}else if(_device_dist == mkt::Distribution::DIST){
		int gpu = get_gpu_by_local_index(index);
		acc_set_device_num(gpu, acc_device_not_host);
		T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
		acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
	}
	acc_wait(0);
}

template<typename T>
T mkt::DArray<T>::get_global(int index) {
  // TODO
}

template<typename T>
void mkt::DArray<T>::set_global(int index, const T& v) {
  // TODO
}

template<typename T>
T& mkt::DArray<T>::operator[](int local_index) {
  	return _data[local_index];
}

template<typename T>
const T& mkt::DArray<T>::operator[](int local_index) const {
  	return _data[local_index];
}

template<typename T>
int mkt::DArray<T>::get_size() const {
  return _size;
}

template<typename T>
int mkt::DArray<T>::get_size_local() const {
  return _size_local;
}

template<typename T>
int mkt::DArray<T>::get_size_gpu() const {
  return _size_gpu;
}

template<typename T>
int mkt::DArray<T>::get_offset() const {
  return _offset;
}

template<typename T>
mkt::Distribution mkt::DArray<T>::get_distribution() const {
  return _dist;
}

template<typename T>
mkt::Distribution mkt::DArray<T>::get_device_distribution() const {
  return _device_dist;
}

template<typename T>
const T* mkt::DArray<T>::get_data() const {
  return _data.data();
}

template<typename T>
T* mkt::DArray<T>::get_data() {
  return _data.data();
}

template<typename T>
T* mkt::DArray<T>::get_device_pointer(int gpu) const{
  return _gpu_data[gpu];
}

template<typename T>
int mkt::DArray<T>::get_gpu_by_local_index(int local_index) const {
	if(_device_dist == mkt::Distribution::COPY){
		return 0;
	}else if(_device_dist == mkt::Distribution::DIST){
		return local_index / _size_gpu;
	}
	else{
		return -1;
	}
}

template<typename T>
int mkt::DArray<T>::get_gpu_by_global_index(int global_index) const {
	// TODO
	return -1;
}
template<typename T, typename R, typename Functor>
void mkt::map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	//
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		const unsigned int gpu_elements = in.get_size_gpu();
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) gang vector async(0)
		for(unsigned int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			out_devptr[i] = f(in_devptr[i]);
		}
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	int offset = in.get_offset();
	int gpu_elements = in.get_size_gpu();
	
	//
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		unsigned int gpu_offset = offset;
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}
		
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) gang vector async(0)
		for(unsigned int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
		}
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	unsigned int gpu_elements = in.get_size_gpu();
	
	//
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		unsigned int gpu_offset = 0;
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset = gpu * gpu_elements;
		}
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) gang vector async(0)
		for(unsigned int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
		}
	}
}

template<typename T, typename Functor>
void mkt::map_in_place(mkt::DArray<T>& a, Functor f){
	unsigned int gpu_elements = a.get_size_gpu();
	
  //
  for(int gpu = 0; gpu < 1; ++gpu){
	acc_set_device_num(gpu, acc_device_not_host);
	f.init(gpu);
	T* devptr = a.get_device_pointer(gpu);
	
	#pragma acc parallel loop deviceptr(devptr) firstprivate(f) gang vector async(0)
  	for(unsigned int i = 0; i < gpu_elements; ++i) {
  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
    	devptr[i] = f(devptr[i]);
  	}
  }
}

template<typename T, typename Functor>
void mkt::map_index_in_place(mkt::DArray<T>& a, Functor f){
	unsigned int offset = a.get_offset();
	unsigned int gpu_elements = a.get_size_gpu();
			  
  	//
  	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* devptr = a.get_device_pointer(gpu);
		
		unsigned int gpu_offset = offset;
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}
		#pragma acc parallel loop deviceptr(devptr) firstprivate(f) gang vector async(0)
	  	for(unsigned int i = 0; i < gpu_elements; ++i) {
	  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
	    	devptr[i] = f(i + gpu_offset, devptr[i]);
	  	}
  	}
}

template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DArray<T>& a, Functor f){
	unsigned int gpu_elements = a.get_size_gpu();
	
  //
  for(int gpu = 0; gpu < 1; ++gpu){
	acc_set_device_num(gpu, acc_device_not_host);
	f.init(gpu);
	T* devptr = a.get_device_pointer(gpu);				
	
	unsigned int gpu_offset = 0;
	if(a.get_device_distribution() == mkt::Distribution::DIST){
		gpu_offset = gpu * gpu_elements;
	}
	#pragma acc parallel loop deviceptr(devptr) firstprivate(f) gang vector async(0)
  	for(unsigned int i = 0; i < gpu_elements; ++i) {
  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
    	devptr[i] = f(i + gpu_offset, devptr[i]);
  	}
  }
}
template<typename T>
mkt::DeviceArray<T>::DeviceArray(const DArray<T>& da)
    : _size(da.get_size()),
      _size_local(da.get_size_local()),
      _size_device(da.get_size_gpu()),
      _offset(da.get_offset()),
      _device_offset(0),
      _dist(da.get_distribution()),
      _device_dist(da.get_device_distribution()) 
{
	_device_data = nullptr;
	for(int i = 0; i < 1; ++i){
		_gpu_data[i] = da.get_device_pointer(i);
	}
}

template<typename T>
mkt::DeviceArray<T>::DeviceArray(const DeviceArray<T>& da)
    : _size(da._size),
      _size_local(da._size_local),
      _size_device(da._size_device),
      _offset(da._offset),
      _device_offset(da._device_offset),
      _dist(da._dist),
      _device_dist(da._device_dist) 
{
	_device_data = da._device_data;
	for(int i = 0; i < 1; ++i){
		_gpu_data[i] = da._gpu_data[i];
	}
}

template<typename T>
mkt::DeviceArray<T>::~DeviceArray(){
}

template<typename T>
void mkt::DeviceArray<T>::init(int gpu) {
	if(_device_dist == Distribution::COPY){
		_device_offset = 0;
	} else {
		_device_offset = _size_device * gpu;
	}
	    
	_device_data = _gpu_data[gpu];
}

template<typename T>
const T& mkt::DeviceArray<T>::get_data_device(int device_index) const {
  return _device_data[device_index];
}


template<typename T>
const T& mkt::DeviceArray<T>::get_data_local(int local_index) const {
  return this->get_data_device(local_index - _device_offset);
}



template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



template<>
void mkt::gather<float>(mkt::DArray<float>& in, mkt::DArray<float>& out){
	in.update_self();
	#pragma omp simd
	for(int counter = 0; counter < in.get_size(); ++counter){
	  out[counter] = in[counter];
	}
	out.update_devices();
}
template<typename T>
void mkt::scatter(mkt::DArray<T>& in, mkt::DArray<T>& out){
	in.update_self();
	#pragma omp simd
	for(unsigned int counter = 0; counter < in.get_size(); ++counter){
	  out.set_local(counter, in.get_local(counter));
	}
	out.update_devices();
}
	
