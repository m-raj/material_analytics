�	Q��90a@Q��90a@!Q��90a@	L�B@#@L�B@#@!L�B@#@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Q��90a@�%�1@1Ӄ�R�Z@A���>e�?I�Ң>��@YW%�}�=*@*	������@2]
&Iterator::Model::FlatMap[0]::Generator�!��*@!��s�J�X@)�!��*@1��s�J�X@:Preprocessing2F
Iterator::Modelq�-�*@!      Y@)�7�-:�?1��3���?:Preprocessing2O
Iterator::Model::FlatMap�͎T߱*@!��Z�X@)�@gҦ�n?1C��S�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9L�B@#@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�%�1@�%�1@!�%�1@      ��!       "	Ӄ�R�Z@Ӄ�R�Z@!Ӄ�R�Z@*      ��!       2	���>e�?���>e�?!���>e�?:	�Ң>��@�Ң>��@!�Ң>��@B      ��!       J	W%�}�=*@W%�}�=*@!W%�}�=*@R      ��!       Z	W%�}�=*@W%�}�=*@!W%�}�=*@JGPUYL�B@#@b �"^
2gradient_tape/Conv42/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2w��Rc^�?!w��Rc^�?"^
2gradient_tape/Conv41/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2�,��F�?!}�q��R�?"^
2gradient_tape/output/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2'f�a��?!�$~�L�?"^
2gradient_tape/Conv12/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2������?!"U,�)��?"^
2gradient_tape/Conv13/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2���!��?!t�d;J��?"^
2gradient_tape/Conv11/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2�䳑�t�?!F�M���?")
Conv41/Conv3DConv3D��]���?!m&V����?"\
1gradient_tape/Conv41/Conv3D/Conv3DBackpropInputV2Conv3DBackpropInputV23!��?!�L���(�?")
Conv42/Conv3DConv3Dl'��� �?!���h}��?"\
1gradient_tape/Conv42/Conv3D/Conv3DBackpropInputV2Conv3DBackpropInputV2������?!��!�>��?Q      Y@Y�Iݗ�V@ad+����W@q�*���c�?ya�Af�ll?"�	
both�Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"t12.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 