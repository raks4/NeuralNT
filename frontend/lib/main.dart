import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'dart:io';

void main() {
  runApp(const NeuralNTApp());
}

class NeuralNTApp extends StatelessWidget {
  const NeuralNTApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuralNT',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
      ),
      home: const Dashboard(),
    );
  }
}

class Dashboard extends StatefulWidget {
  const Dashboard({super.key});

  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  int _selectedIndex = 0;
  String _architectureText = "";

  final _lrController = TextEditingController(text: "0.001");
  final _batchController = TextEditingController(text: "32");
  final _sizeController = TextEditingController(text: "32");
  final _epochController = TextEditingController(text: "10");

  String _selectedLoss = "CrossEntropyLoss";
  String _selectedOptimizer = "Adam";
  int _selectedChannels = 3;
  PlatformFile? _selectedFile;
  bool _isTraining = false;
  String _trainingLogs = "";

  PlatformFile? _predictionFile;
  String _predictionResult = "";
  bool _isPredicting = false;

  final String baseUrl = Platform.isAndroid ? "http://10.0.2.2:8000" : "http://localhost:8000";

  @override
  void initState() {
    super.initState();
    _fetchArchitecture();
  }

  Future<void> _fetchArchitecture() async {
    try {
      final response = await http.get(Uri.parse("$baseUrl/architecture"));
      if (response.statusCode == 200) {
        setState(() {
          _architectureText = json.decode(response.body)['text'];
        });
      }
    } catch (e) {
      debugPrint("Error fetching arch: $e");
    }
  }

  Future<void> _addLayer(String type, String inDim, String outDim) async {
    try {
      final response = await http.post(
        Uri.parse("$baseUrl/add_layer"),
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "layer_type": type,
          "in_dim": inDim,
          "out_dim": outDim,
        }),
      );
      if (response.statusCode == 200) {
        _fetchArchitecture();
      }
    } catch (e) {
      debugPrint("Error adding layer: $e");
    }
  }

  Future<void> _startTraining() async {
    if (_selectedFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select a dataset file first")),
      );
      return;
    }

    setState(() {
      _isTraining = true;
      _trainingLogs = "Starting training...\n";
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse("$baseUrl/train"));
      request.fields['loss_name'] = _selectedLoss;
      request.fields['opt_name'] = _selectedOptimizer;
      request.fields['lr'] = _lrController.text;
      request.fields['batch_size'] = _batchController.text;
      request.fields['image_size'] = _sizeController.text;
      request.fields['epochs'] = _epochController.text;
      request.fields['num_channels'] = _selectedChannels.toString();
      request.fields['generate_animation'] = "false";

      request.files.add(await http.MultipartFile.fromPath(
        'dataset',
        _selectedFile!.path!,
      ));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        setState(() {
          _trainingLogs += "\nSuccess!\n${data['logs']}";
        });
      } else {
        setState(() {
          _trainingLogs += "\nError: ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        _trainingLogs += "\nException: $e";
      });
    } finally {
      setState(() {
        _isTraining = false;
      });
    }
  }

  Future<void> _runPrediction() async {
    if (_predictionFile == null) return;

    setState(() {
      _isPredicting = true;
      _predictionResult = "Running inference...";
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse("$baseUrl/predict"));
      request.fields['image_size'] = _sizeController.text;
      request.files.add(await http.MultipartFile.fromPath(
        'image',
        _predictionFile!.path!,
      ));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        setState(() {
          _predictionResult = "Result: ${data['prediction']}\nConfidence: ${data['confidence']}";
        });
      } else {
        setState(() {
          _predictionResult = "Error: ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        _predictionResult = "Exception: $e";
      });
    } finally {
      setState(() {
        _isPredicting = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("NeuralNT Dashboard"),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _fetchArchitecture,
          )
        ],
      ),
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (int index) {
              setState(() {
                _selectedIndex = index;
              });
            },
            labelType: NavigationRailLabelType.all,
            destinations: const [
              NavigationRailDestination(icon: Icon(Icons.build), label: Text('Build')),
              NavigationRailDestination(icon: Icon(Icons.model_training), label: Text('Train')),
              NavigationRailDestination(icon: Icon(Icons.remove_red_eye), label: Text('Test')),
            ],
          ),
          const VerticalDivider(thickness: 1, width: 1),
          Expanded(
            child: _selectedIndex == 0 ? _buildTab() : (_selectedIndex == 1 ? _trainTab() : _predictTab()),
          ),
        ],
      ),
    );
  }

  Widget _buildTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Architecture", style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black26,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.deepPurple),
            ),
            child: Text(
              _architectureText.isEmpty ? "No layers added yet." : _architectureText,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 16),
            ),
          ),
          const SizedBox(height: 20),
          const Divider(),
          const SizedBox(height: 10),
          Text("Add New Layer", style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 10),
          Wrap(
            spacing: 10,
            children: [
              _layerButton("Linear"),
              _layerButton("Conv2d"),
              _layerButton("ReLU"),
              _layerButton("MaxPool2d"),
              _layerButton("Flatten"),
            ],
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: () async {
              await http.post(Uri.parse("$baseUrl/reset"));
              _fetchArchitecture();
            },
            icon: const Icon(Icons.delete_forever),
            label: const Text("Reset All Layers"),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red.withOpacity(0.2),
              foregroundColor: Colors.red,
            ),
          )
        ],
      ),
    );
  }

  Widget _layerButton(String type) {
    return ActionChip(
      label: Text(type),
      onPressed: () => _showAddDialog(type),
    );
  }

  void _showAddDialog(String type) {
    if (type == "ReLU" || type == "Flatten") {
       _addLayer(type, "", "");
       return;
    }
    final TextEditingController inCtrl = TextEditingController();
    final TextEditingController outCtrl = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text("Add $type Layer"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: inCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: "Input Dimension")),
            TextField(controller: outCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: "Output Dimension")),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("Cancel")),
          ElevatedButton(
            onPressed: () {
              _addLayer(type, inCtrl.text, outCtrl.text);
              Navigator.pop(context);
            },
            child: const Text("Add"),
          )
        ],
      ),
    );
  }

  Widget _trainTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Train Model", style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 20),
          DropdownButtonFormField<String>(
            value: _selectedLoss,
            decoration: const InputDecoration(labelText: "Loss Function"),
            items: ["CrossEntropyLoss", "MSELoss"].map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
            onChanged: (v) => setState(() => _selectedLoss = v!),
          ),
          const SizedBox(height: 10),
          DropdownButtonFormField<String>(
            value: _selectedOptimizer,
            decoration: const InputDecoration(labelText: "Optimizer"),
            items: ["Adam", "SGD"].map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
            onChanged: (v) => setState(() => _selectedOptimizer = v!),
          ),
          const SizedBox(height: 10),
          TextField(controller: _lrController, decoration: const InputDecoration(labelText: "Learning Rate")),
          TextField(controller: _batchController, decoration: const InputDecoration(labelText: "Batch Size")),
          TextField(controller: _sizeController, decoration: const InputDecoration(labelText: "Image Size")),
          TextField(controller: _epochController, decoration: const InputDecoration(labelText: "Epochs")),
          const SizedBox(height: 20),
          // Changed Row to Wrap to prevent overflow error
          Wrap(
            spacing: 10,
            runSpacing: 10,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: () async {
                  FilePickerResult? result = await FilePicker.platform.pickFiles();
                  if (result != null) setState(() => _selectedFile = result.files.first);
                },
                icon: const Icon(Icons.file_open),
                label: Text(_selectedFile == null ? "Select Dataset (.zip)" : "File: ${_selectedFile!.name}"),
              ),
              if (_isTraining) const CircularProgressIndicator()
              else ElevatedButton(
                onPressed: _startTraining,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.green.withOpacity(0.2), foregroundColor: Colors.green),
                child: const Text("START TRAINING"),
              ),
            ],
          ),
          const SizedBox(height: 20),
          const Divider(),
          const Text("Training Logs:", style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(8),
            color: Colors.black45,
            child: Text(_trainingLogs, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
        ],
      ),
    );
  }

  Widget _predictTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.image_search, size: 80, color: Colors.deepPurple),
          const SizedBox(height: 20),
          Text("Test Your Model", style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 10),
          const Text("Upload a single image to see what the model thinks it is.", textAlign: TextAlign.center),
          const SizedBox(height: 30),

          if (_predictionFile != null) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.file(File(_predictionFile!.path!), height: 150),
            ),
            const SizedBox(height: 10),
          ],

          ElevatedButton.icon(
            onPressed: () async {
              // CHANGED: Using FileType.any to avoid the blank Photo Picker bug
              FilePickerResult? result = await FilePicker.platform.pickFiles();
              if (result != null) setState(() => _predictionFile = result.files.first);
            },
            icon: const Icon(Icons.photo_library),
            label: const Text("Pick an Image"),
          ),
          const SizedBox(height: 20),

          if (_isPredicting) const CircularProgressIndicator()
          else ElevatedButton(
            onPressed: _runPrediction,
            style: ElevatedButton.styleFrom(backgroundColor: Colors.orange.withOpacity(0.2), foregroundColor: Colors.orange),
            child: const Text("RUN PREDICTION"),
          ),

          const SizedBox(height: 30),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: Colors.black45, borderRadius: BorderRadius.circular(8)),
            child: Text(
              _predictionResult.isEmpty ? "No result yet." : _predictionResult,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.orangeAccent),
            ),
          ),
        ],
      ),
    );
  }
}
